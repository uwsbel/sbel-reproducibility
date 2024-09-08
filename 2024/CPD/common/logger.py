

class Logger():
    def __init__(self,filename="",display=True,write=True):
        self.filename=filename
        self.display=display
        self.write=write

        if(self.write):
            self.file_out = open(filename,"w")

    def log(self,line):

        if(self.display):
            print(line)

        if(self.write):
            self.file_out.write(line)
            self.file_out.write("\n")

    def close(self):
        if(self.write):
            self.file_out.close()
