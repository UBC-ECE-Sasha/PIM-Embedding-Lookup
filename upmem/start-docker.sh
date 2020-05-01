#!/bin/bash 

# Start docker
sudo systemctl start docker

sudo docker run -it --rm -v /home/niloofar/upmem:/root/upmem upmem_sdk_base /bin/bash
