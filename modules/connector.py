import jaydebeapi
import paramiko

from hdfs import *

def dbConnection():
    # connection to cubrid DB
    cubrid_ip = "0.0.0.0"
    port = ""
    conn = jaydebeapi.connect(
        "cubrid.jdbc.driver.CUBRIDDriver",
        f"jdbc:cubrid:{cubrid_ip}:{port}:DB_name:::?charset=utf-8?",
        ['id','passwd'],
        "path/to/java.jar"
    )
    return conn

def hdfsConnection():
    # connection to hdfs
    name_node_ip1 = "0.0.0.0"
    name_node_ip2 = "0.0.0.0"
    port = "50070"
    try:
        hdfs = Client(f"http://{name_node_ip1}:{port}",proxy='name')
        hdfs.status("/")
    except:
        try:
            hdfs = Client(f"http://{name_node_ip2}:{port}",proxy='name')
            hdfs.status("/")
        except HdfsError:
            hdfs = 'None'
        return hdfs
    return hdfs

def nasConnection():
    # connection to ssh
    ssh_ip = '0.0.0.0'
    port = ''
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname = ssh_ip,port=port, username='name',password='passwd')
    return ssh