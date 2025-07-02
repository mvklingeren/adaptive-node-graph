## INSTALL AND START DOCKER
# Add Docker's official GPG key
# mkdir -p /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# # Add Docker repository
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# # Update package index again
# apt-get update

# # Install Docker
# apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# systemctl start docker


## INSTALL NVM AND NODE
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh > install.sh
chmod +x install.sh
./install.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

nvm install 20
nvm use 20



## CLONE AND INSTALL ADAPT / GENERATE CUDA KERNEL
git clone https://github.com/mvklingeren/adaptive-node-graph
mv adaptive-node-graph adapt
cd adapt
git checkout feature/cuda-graph
npm install
npm run generate-cu:shakespear

make