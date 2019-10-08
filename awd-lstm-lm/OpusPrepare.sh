cd /home/work

#  $HOME is not the same as ~ !!!!

# Installing pyenv and putting it in the path
curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
echo "HOME is $HOME"
echo 'export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
' > $HOME/.bashrc

# Installing python 3.5 and making it default
source $HOME/.bashrc
pyenv install 3.5.2
pyenv local 3.5.2
# Note: when we exit the script, environments go away and we need to re-source ~/.bashrc and re-run pyenv local 3.5.2

# Installing numpy and PyTorch
pip install numpy==1.14
pip install torch

apt-get install unzip  # Some machines seem not to have it?

# Downloading the data
sh ./getdata.sh

