import os
import glob
import datetime
import jaydebeapi
import traceback

import pandas as pd

from absl import flags, app
from absl.flags import FLAGS
from modules.connector import hdfsConnection, dbConnection

flags.DEFINE_string("FROM","/user/input/","hadoop download directory path")
flags.DEFINE_string("HOME_PATH","/data/","local directory path to download")
flags.DEFINE_string("todo","","dir name todo")

def main(_argv):
    from_path = FLAGS.FROM
    home_path = FLAGS.HOME_PATH
    todo_dir = FLAGS.todo
    
    client = hdfsConnection()

    if client is None:
        print("hdfs connection error")
        return

    try:
        if todo_dir not in client.list(from_path):
            print(f"{from_path}/{todo_dir} not in hadoop server")
            return
        total_files = client.list(f'{from_path}/{todo_dir}/')
        print("hadoop connnected")
    except:
        print(traceback.format_exc())
        return
    
    save_dir_path = f"{home_path}/{todo_dir}"
    os.makedirs(save_dir_path, exist_ok=True)

    try:
        conn = dbConnection()
        curr = conn.cursor()

        sql = f"""
        SELECT *
        FROM meta_tbl
        WHERE hdfs_stgr_path LIKE '/user/input/%{todo_dir}%'
        """
        curr.execute(sql)
        out_data = curr.fetchall()
        columns = [i[0] for i in curr.description]

        df = pd.DataFrame(data=out_data, columns=columns)
        curr.close()
        print("cubrid connected")
    except:
        print(traceback.format_exc())
        return

    size_dict = {f"{i}_{j}_{k}.{l}":m for i,j,k,l,m in zip(df.sn, df.data_sn, df.file_nm, df.ext,df.file_size)}

    # Download start
    try:
        for ind in range(len(df)):
            now_data = df.loc[ind]
            hdfs_file_path = now_data.hdfs_stgr_path
            save_file_path = f"{save_dir_path}/{hdfs_file_path.split('/')[-1]}"

            sn = now_data.sn
            data_sn = now_data.data_sn

            client.download(hdfs_file_path, save_file_path, overwrite=True)

            # Download File check
            if os.path.isfile(save_file_path):
                if os.path.getsize(save_file_path) != now_data.file_size:
                    print(f"[size miss] {save_file_path}")
                    df = df.drop(index=ind)
                    os.remove(save_file_path)
            else:
                print(f"download fail {save_file_path}")

        os.makedirs(f"{home_path}/cubrid/{todo_dir}",exist_ok=True)
        df.to_csv(f"{home_path}/cubrid/{todo_dir}/meta_tbl.csv",index=False)

        print(f"[Finish] download({todo_dir}) : {len(df)}\n")

    except:
        print(traceback.format_exc())


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
