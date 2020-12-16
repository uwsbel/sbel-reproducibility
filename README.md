# The "public-metadata" repo
The repo contains assets, obj files, json files, scripts, Chrono models, etc., needed to generate results SBEL members report in papers, tech reports, presentations, etc.
This repo is organized by year of when the metadata was first added to the repo.

## Info for SBEL members
- Subfolders are meant to contain information associated with a certain paper, tech report, presentation, etc. Feel free to use subfolders in subfolders.
- Please include a readme.md file in the high level subfolder to describe the content of the directroy along with where the material was used (which paper/tech report/etc.).
- **Important:** If the metadata is meant to be used in conjunction with Chrono, please provide a commit ID/SHA1 - hash value that the results reported in your paper/tech report/etc. were generated with
- Please avoid dropping large files here, particularly so if they're non-ascii. For large files that are available elsewhere, provide a link (in your readme.md) that can be used to download the file. This applies, for instance, to large pics, movies, etc.
- When adding data, scripts, assets, models, etc., please take the long view. The metadata that you provide will likely be used for years to come. 
- Style issue: When referencing in your manuscript this metadata repo, say for TR-2020-02, please define in SBEL's **BibFiles** repo, under **refsSBELspecific.bib**, an entry like this:
*@misc{TR-2020-02metadata,
author = {Hu, Wei and Serban, Radu and Negrut, Dan},
title = {{TR-2020-02 Public Metadata}},
note              = {{Simulation-Based Engineering Laboratory, University of Wisconsin-Madison}},
year              = {2020},
howpublished      = {\url{https://github.com/uwsbel/public-metadata/tree/master/2020/TR-2020-02}}
}*
In order to maintain consistency accross docs for all our references, please copy/paste/edit the sample above to fit your needs when dropping in **refsSBELspecific.bib**.
