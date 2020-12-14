# Experimenentation with MLFlow

In this repo I explore how to set up an MLFlow Tracking server on a remote machine, and how to log machine learning training/runs/experiments to it. 

## Setting up an MLFlow tracking server

### 1. Configure an EC2 instance in AWS
If you don't already have one, configure an Ubuntu EC2 instance on AWS, of whatever kind you wish. Make sure to set up SSH to it. In the security groups for the EC2 instance, make sure at least your IP address has incoming access to port 80 on TCP (or alternatively make port 80 publically available -- we will put a password on it later anyhow). Once you're done, SSH into the machine. 

Finally, ensure that the role the instance assumes has full access to S3 (can be configured in the role permissions and changed later if you get it wrong).

### (Optional) 1.2 Create an S3 bucket where you'll store the artifacts
Go to S3, create a bucket with no public access

### 2. Install dependencies

```bash
# Get Python & pip
sudo apt-get install python3.7
echo "export PATH=\"/home/ubuntu/.local/bin:\$PATH\"" >> ~/.bashrc && source ~/.bashrc

mkdir downloads
cd downloads/
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 python get-pip.py #python3 just in case 

# Install MLFlow and others
sudo pip3 install mlflow
sudo pip3 install -U python-dateutil==2.6.1
sudo pip3 install boto3

# Get and start NGINX
sudo apt-get install nginx
sudo service nginx start
sudo apt-get install apache2-utils -y

# AWS CLI
sudo apt-get install awscli
aws configure
aws sts get-caller-identity # just to check what the role and user of the machine are
```

### 3. Configure NGINX and password authentication

```bash
sudo htpasswd -c /etc/nginx/.htpasswd <my_user>
#... provide desired password
```

Now, you'll have to configure a "sites-available" site for NGINX, create a symlink to the sites-enabled folder, and finally modify that so that any incoming traffic to port `80` is forwarded to port `5000`, which is where the MLFLow tracking server will be setup. 


```bash
cp /etc/nginx/sites-available/default /etc/nginx/sites-available/auth.conf # make a copy to edit
vim /etc/nginx/sites-available/auth.conf
```

Now, add the following lines in the file: 

```bash
    # Add index.php to the list if you are using PHP
    index index.html index.htm index.nginx-debian.html; # already there

    server_name _; # already there

    #### THIS IS THE NEW STUFF TO INSERT!!! ####
    location / { 
        proxy_pass http://localhost:5000/;
        auth_basic "Restricted Content";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
```

Finally, create a symlink for this file in the `sites-enabled` directory, as that's where NGINX reads from

```bash
sudo ln -s /etc/nginx/sites-available/auth.conf /etc/nginx/sites-enabled/auth.conf
sudo rm /etc/nginx/sites-enabled/default # remove the default one as we don't need it anymore
```

### 4. Start the server
To set the MLflow tracking server running (in the background), execute the following command: 

```bash
mkdir mlflow
nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://<your-bucket-name>/ --host 0.0.0.0 > mlflow/logs/mlflow_tserver.log 2>&1 &
```

Explanation: 
- `--backend-store-uri sqlite:///mlflow.db`: The tracking backend will store information in a local sqlite database. You don't need to install sqlite separately, as it comes with `mlflow`. You will be able to find the file where the database stores your tracking metadata under `mlflow.db`
- `--default-artifact-root s3://<your-bucket-name>/`: This is where you tell `mlflow` that it should store your artifacts in this location in S3. It doesn't have to be a bucket, it can also be bucket + key. 
- `--host 0.0.0.0` : which host to run the tracking server on. It will take port 5000 by default (hence the port forwarding we did with NGINX)
- `> mlflow/logs/mlflow_tserver.log 2>&1`: this means redirect all output from this process into a logfile under the specified back, both stout plus errors. 
- `&` : run in the background 

You can also set it running in a `tmux` session if you prefer. 

### Enjoy!
Now, if you access `http://<ec2-dns-address>/` you will see the MLFlow UI! 

