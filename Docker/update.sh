#!/bin/bash
# Select right directory
cd "/home/"

# Remove existing
rm -r ./lcer
rm -r ./shadowhand-gym

# Clone new
git clone https://github.com/szahlner/lcer.git
git clone https://github.com/szahlner/shadowhand-gym.git
pip3 install -e shadowhand-gym

# Set permissions and prepare runs
cd "/home/lcer/scripts/"
chmod +x "./prepare_experiments.sh"
./prepare_experiments.sh