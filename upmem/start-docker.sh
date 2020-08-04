#!/bin/bash 

# Start docker
sudo systemctl start docker

sudo docker run -it --rm -v /home/niloofar/PIM-Embedding-Lookup/upmem:/root/upmem john_upmem_sdk /bin/bash
