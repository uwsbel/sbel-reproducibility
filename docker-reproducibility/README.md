# **docker-reproducibility** subfolder

This subfolder contains several sub-subfolders:
  * The subfolder *docker-scripts* contains stock files that are to be used as a starting point to produce a new image in the Docker Hub registry. This image would be employed by a user to quickly deploy a Docker container in order to reproduce our results (reported in journals, conference proceedings, technical reports, etc.)

  * Any other subfolder should be named according to a tag associated with the corresponding **uwsbel/reproducibility image** in Docker Hub. For instance, if Luning Fang has a 2023 paper on which she's the first author, the tag (and therefore the name of the sub-subfolder herein) should be *2023Fang-continuousContact*. The naming convention observed: name of year, last name of author, and super brief two or three word description of the image. The sub-subfolder *2023Fang-continuousContact* would contain the scripts and metadata needed to generate the Docker Hub image uwsbel/reproducibility:2023Fang-continuousContact* (note that 2023Fang-continuousContact is a tag on the uwsbel/reproducibility image). Then, anybody from outside the lab can pull the Docker image via 
      * \>\> docker pull uwsbel/reproducibility:2023Fang-continuousContact

The underlying idea is that any journal manuscript, conference contribution, technical report, thesis, indepedent study that reports results should have a *uwsbel/reproducibility:tag-name-here* image in the Docker Hub registry. The subfolder *docker-scripts* has helper scripts that you need to slightly massage. These massaged scripts are to be included in a subfolder *2023Fang-continuousContact* (in the example above) and an image tagged like *2023Fang-continuousContact* should be present in the Docker Hub registry.
