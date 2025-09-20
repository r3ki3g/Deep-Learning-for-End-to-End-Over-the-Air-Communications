from scipy.io import loadmat
import pandas as pd
data = loadmat(r"D:\Engineering\ENTC\Final Year Project\Project\Deep-Learning-for-End-to-End-Over-the-Air-Communications\Lathika\dataset\received_iq.mat")

print(data.keys())
x=data["received_iq"]
df=pd.DataFrame(x)

df.to_csv(r"D:\Engineering\ENTC\Final Year Project\Project\Deep-Learning-for-End-to-End-Over-the-Air-Communications\Lathika\dataset\recieved_iq.csv")


