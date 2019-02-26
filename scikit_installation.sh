!#/bin/bash

echo "=====================================================================================================================================";
echo "===	This script will install all necessary packages for scikit-learn (Python Library of Machine Learning Algorithms.	===";
echo "=====================================================================================================================================";

# Installs packages: buid-essential, python-dev, python-numpy, python-setuptools, python-scipy, libatlas-dev
sudo apt-get install -y build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev;
sudo -H pip install numpy;
sudo -H pip install scipy;
sudo -H pip install scikit-learn;

# Installs matplotlib
sudo apt-get install  -y python-matplotlib;
sudo apt-get install  -y libpng-dev libjpeg8-dev libfreetype6-dev

# Installs scikit-learn package
sudo -H pip install scikit-learn;

# Installs IPython Notebook package
sudo apt-get install  -y ipython-notebook;
sudo -H pip install tornado;
sudo -H pip install pyzmq;

