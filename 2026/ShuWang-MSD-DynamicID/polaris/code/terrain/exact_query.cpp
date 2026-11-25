// Terrain query helper for the uneven road of the Polaris model.
// It replaces the terrain object that the tires query by one that answers height, normal and friction
// from the triangles of the RigidTerrain height-map mesh, in double precision (a uniform grid of bins
// finds the triangle). Points outside the mesh are passed to the original RigidTerrain.
// Built as a DLL against the Chrono headers and libraries of the pychrono build; see BUILD.md.
#include <chrono_vehicle/ChTerrain.h>
#include <chrono_vehicle/terrain/RigidTerrain.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using chrono::ChVector;
using chrono::vehicle::ChTerrain;
struct Face {
    double x0,y0,x1,y1,x2,y2,z0,z1,z2,den,nx,ny,nz;
};
struct Query {
    std::vector<Face> faces;
    std::vector<std::vector<int>> bins;
    double xmin,xmax,ymin,ymax,mu;
    int nx,ny;
    uint64_t hc=0,nc=0,fc=0,outside=0,inside_miss=0,hfallback=0,nfallback=0,ffallback=0;
    std::shared_ptr<ChTerrain> fallback;
    Query(const double* v,size_t nv,const int32_t* idx,size_t nf,double friction):mu(friction) {
        if(nv<3 || nf<1 || !std::isfinite(mu) || mu<=0) throw std::runtime_error("invalid mesh");
        xmin=xmax=v[0];ymin=ymax=v[1];
        for(size_t i=0;i<nv;i++) {
            for(int k=0;k<3;k++) if(!std::isfinite(v[3*i+k])) throw std::runtime_error("nonfinite vertex");
            xmin=std::min(xmin,v[3*i]);xmax=std::max(xmax,v[3*i]);
            ymin=std::min(ymin,v[3*i+1]);ymax=std::max(ymax,v[3*i+1]);
        }
        if(xmax<=xmin || ymax<=ymin) throw std::runtime_error("degenerate XY bounds");
        nx=ny=std::max(4,int(std::ceil(std::sqrt(double(nf)/2.))));bins.resize(nx*ny);
        for(size_t j=0;j<nf;j++) {
            const double* p[3];
            for(int k=0;k<3;k++){int n=idx[3*j+k];if(n<0 || size_t(n)>=nv)throw std::runtime_error("bad triangle index");p[k]=v+3*n;}
            Face f={p[0][0],p[0][1],p[1][0],p[1][1],p[2][0],p[2][1],p[0][2],p[1][2],p[2][2]};
            double ax=p[1][0]-p[0][0],ay=p[1][1]-p[0][1],az=p[1][2]-p[0][2];
            double bx=p[2][0]-p[0][0],by=p[2][1]-p[0][1],bz=p[2][2]-p[0][2];
            f.nx=ay*bz-az*by;f.ny=az*bx-ax*bz;f.nz=ax*by-ay*bx;
            double norm=std::sqrt(f.nx*f.nx+f.ny*f.ny+f.nz*f.nz);
            if(norm<=0 || std::abs(f.nz)<1e-16)throw std::runtime_error("non-heightfield triangle");
            if(f.nz<0)norm=-norm;f.nx/=norm;f.ny/=norm;f.nz/=norm;
            f.den=(f.y1-f.y2)*(f.x0-f.x2)+(f.x2-f.x1)*(f.y0-f.y2);
            faces.push_back(f);
            int a=ix(std::min({f.x0,f.x1,f.x2})),b=ix(std::max({f.x0,f.x1,f.x2}));
            int c=iy(std::min({f.y0,f.y1,f.y2})),d=iy(std::max({f.y0,f.y1,f.y2}));
            for(int yy=c;yy<=d;yy++)for(int xx=a;xx<=b;xx++)bins[yy*nx+xx].push_back(int(j));
        }
    }
    int ix(double x)const{return std::max(0,std::min(nx-1,int(std::floor((x-xmin)/(xmax-xmin)*nx))));}
    int iy(double y)const{return std::max(0,std::min(ny-1,int(std::floor((y-ymin)/(ymax-ymin)*ny))));}
    int lookup(double x,double y,double& height) {
        if(!std::isfinite(x) || !std::isfinite(y) || x<xmin-1e-12 || x>xmax+1e-12 || y<ymin-1e-12 || y>ymax+1e-12) {outside++;height=-1e10;return -1;}
        for(int j:bins[iy(y)*nx+ix(x)]) {
            const Face& f=faces[j];
            double a=((f.y1-f.y2)*(x-f.x2)+(f.x2-f.x1)*(y-f.y2))/f.den;
            double b=((f.y2-f.y0)*(x-f.x2)+(f.x0-f.x2)*(y-f.y2))/f.den;
            double c=1-a-b;
            if(a>=-1e-10 && b>=-1e-10 && c>=-1e-10) {height=a*f.z0+b*f.z1+c*f.z2;return j;}
        }
        inside_miss++;height=-1e10;return -1;
    }
};
struct ExactNativeTerrain:ChTerrain {
    std::shared_ptr<Query> q;
    explicit ExactNativeTerrain(std::shared_ptr<Query> x):q(x){}
    double GetHeight(const ChVector<>& loc)const override{
        q->hc++;double h;int j=q->lookup(loc.x(),loc.y(),h);
        if(j<0){q->hfallback++;return q->fallback->GetHeight(loc);}return h;
    }
    ChVector<> GetNormal(const ChVector<>& loc)const override{
        q->nc++;double h;int j=q->lookup(loc.x(),loc.y(),h);
        if(j<0){q->nfallback++;return q->fallback->GetNormal(loc);}
        const auto& f=q->faces[j];return ChVector<>(f.nx,f.ny,f.nz);
    }
    float GetCoefficientFriction(const ChVector<>& loc)const override{
        q->fc++;double h;int j=q->lookup(loc.x(),loc.y(),h);
        if(j<0){q->ffallback++;return q->fallback->GetCoefficientFriction(loc);}return float(q->mu);
    }
};
static thread_local std::string error_text;
extern "C" {
__declspec(dllexport) void* install_exact(void* swig_shared_ptr,void* rigid_shared_ptr,const double* vertices,size_t nv,const int32_t* faces,size_t nf,double mu) {
    try {
        auto terrain=*static_cast<std::shared_ptr<ChTerrain>*>(swig_shared_ptr);
        if(!terrain)throw std::runtime_error("null terrain");
        auto q=std::make_shared<Query>(vertices,nv,faces,nf,mu);
        if(!rigid_shared_ptr)throw std::runtime_error("original native RigidTerrain required");
        q->fallback=*static_cast<std::shared_ptr<chrono::vehicle::RigidTerrain>*>(rigid_shared_ptr);
        if(!q->fallback)throw std::runtime_error("null native fallback");
        *static_cast<std::shared_ptr<ChTerrain>*>(swig_shared_ptr)=std::make_shared<ExactNativeTerrain>(q);
        error_text.clear();return q.get();
    }catch(const std::exception& e){error_text=e.what();return nullptr;}
}
__declspec(dllexport) const char* exact_error(){return error_text.c_str();}
__declspec(dllexport) void exact_counts(void* handle,uint64_t* out){auto q=static_cast<Query*>(handle);out[0]=q->hc;out[1]=q->nc;out[2]=q->fc;out[3]=q->outside;out[4]=q->inside_miss;out[5]=q->hfallback;out[6]=q->nfallback;out[7]=q->ffallback;}
}
