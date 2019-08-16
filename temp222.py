# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np



MyList = [1,2,3,4,5,6,7,8,9,10]

list(filter(lambda x: x % 3 == 0, MyList))

list(map(lambda x: x * 2, MyList))


from functools import reduce ##二元计算函数
reduce(lambda x, y: x + y, MyList) #求和计算

import os
print(os.getcwd())

os.chdir('E:\python documents')
print(os.getcwd())

MyArray1 = np.arange(1,20) 
MyArray1

range(1,10,2)

list(range(1,10,2))

MyArray2=np.array([1,2,3,4,3,5])
MyArray2

MyArray3=np.zeros((5,5)) 
MyArray3

MyArray4=np.ones((5,5))
MyArray4

np.full((3,5),2)

##设置随机数种子，方便日后复原结果
rand=np.random.RandomState(1)

MyArray5=rand.randint(0,100,[3,5])   #随机取数的上限下限都是在哪里
MyArray5

MyArray6=np.zeros([4,5],dtype=np.int)
MyArray6

myArray=np.array(range(0,10))

print("myArray=",myArray)
print("myArray[1:9:2]=",myArray[1:9:2]) 
print("myArray[:9:2]=",myArray[:9:2]) 
print("myArray[::2]=",myArray[::2]) 
print("myArray[::]=",myArray[::])   
print("myArray[:8:]=",myArray[:8:])  
print("myArray[:8]=",myArray[0:8])  
print("myArray[4::]=",myArray[4::])  
print("myArray[9:1:-2]=",myArray[9:1:-2]) 
print("myArray[::-2]=",myArray[::-2])  
##初学者容易犯的错误，导致数据维度过多的问题 
print("myArray[[2,5,6]]=",myArray[[2,5,6]]) 
print("myArray[myArray>5]=",myArray[myArray>5]) 

MyArray7=np.arange(1,21)  
MyArray7

MyArray7.shape

MyArray8=MyArray7.reshape(4,5)  
MyArray8

#排列顺序按照列进行排列
MyArray8=MyArray8.swapaxes(0,1)
MyArray8
#将多行数据转化成一行的数据
MyArray8.flatten()

MyArray8.tolist()

MyArray8.astype(np.float)

np.rank(MyArray5)  

np.ndim(MyArray5)

np.shape(MyArray5)

MyArray5.shape

MyArray5.size

type(MyArray5)  

MyArray5*10

x=np.array([11,12,13,14,15,16,17,18])
x1,x2,x3=np.split(x,[3,5]) 
print(x1,x2,x3)

upper,lower=np.vsplit(MyArray5.reshape(5,3),[1]) ##这个是指下限区间取不到
print("上半部分为\n",upper)
print("\n\n下半部分为\n",lower)


np.concatenate((lower,upper),axis=0)

np.vstack([upper,lower])  ##倒着堆叠

np.hstack([upper,lower])  ##水平堆叠

np.add(MyArray5,1)       ##做的是矩阵加法 每个元素都相加

np.zeros(10,dtype="int16")

np.zeros(10,dtype="float")

a1=np.array([1,2,3,None])
a1

a1=np.array([1,2,3,None,np.nan])
a1

myArray1=np.array([11,12,13,14,15,16,17,18])
np.delete(myArray1,2)

np.insert(myArray1,1,88) #数组、位置、值

##缺失值处理
np.isnan(myArray1)

np.any(np.isnan(myArray1))

np.all(np.isnan(myArray1))

MyArray=np.array([1,2,3,np.nan])
np.nansum(MyArray)   #连着nan也放在一起进行相加



##这个是表示广播原则
A1=np.array(range(1,10)).reshape([3,3])
A1

A2=np.array([10,10,10])
A2

A1+A2
##列数相同才可以广播相加
A3=np.arange(10).reshape(2,5)    
A3

A4=np.arange(16).reshape(4,4)
A4

A3+A4

##ndarray的排序原则
myArray=np.array([11,18,13,12,19,15,14,17,16])
myArray

np.sort(myArray)

np.argsort(myArray)

MyArray=np.array([[21, 22, 23, 24,25],
       [35,  34,33, 32, 31],
       [ 1, 2,  3, 100, 4]])

np.sort(MyArray,axis=1) 

np.sort(MyArray,axis=0)

import pandas as pd
mySeries1=pd.Series(data = [11,12,13,14,15,16,17],index=["a","b","c","d","e","f","g"]) 
mySeries1

mySeries2=pd.Series([10], index=["a","b","c","d","e","f","g"]) 
mySeries2

mySeries4=pd.Series([21,22,23,24,25,26,27], index=["a","b","c","d","e","f","g"]) 
mySeries4.index

mySeries4.values  

mySeries4['b']

mySeries4[["a","b","c"]] 

mySeries4["a":"d"] 

mySeries4[1:4:2]

mySeries4

"c" in mySeries4

mySeries4=pd.Series([21,22,23,24,25,26,27], index=["a","b","c","d","e","f","g"]) 
mySeries5=mySeries4.reindex(index=["b","c","a","d","e","g","f"])
mySeries5 

##关于dataframe的相关操作
import numpy as np

df2=pd.DataFrame(np.arange(10).reshape(2,5))
df2
df2.index

df2.index.size

df2.columns

df2.columns.size

df2 = pd.read_csv('C:/Users/Administrator/Desktop/数据分析课件/PythonFromDAToDS-master/DataSets/bc_data.csv')
df2.shape

df2=df2[["id","diagnosis","area_mean"]]  ##取三列，然后查看这三列

df2.head()

df2.shape
df2.index.size
df2.columns
df2.columns.size

##引用行或者列
df2['id'].head()
df2.id.head()

df2["id"][2]

df2.id[2]

df2["id"][[2,4]]

##第二种方法我们可以称之为iloc方法
df2.loc[1,"id"] 
df2.iloc[1,0]
df2.ix[1,"id"]
df2.ix[[1,5],["id"]]
df2.ix[1:5,["id"]]
df2[["area_mean","id"]].head()  ##可以调整列输出新的数据框

###index操作
df2.index
df2.columns
df2["id"].head()
df2.reindex(index=["1","2","3"],columns=["1","2","3"])
df2.head()

df2.reindex(index=[2,3,1], columns=["diagnosis","id","area_mean"])##调整列的位置

df3=df2.reindex(index=[2,3,1], columns=["diagnosis","id","area_mean","MyNewColumn"],fill_value=100)
df3

df2=df2[["id","diagnosis","area_mean"]]
df2.head()
df2.drop([2]).head()   ##删除行
df2.head()


df2.drop([3,4],axis=1, inplace=True) ##关于行的删除放入列表形式,关于列的删除放入列名称
df2.drop('id', axis=1, inplace=True)  ##是关于是否更新行索引的办法

del df2["area_mean"] 
df2.head()

df2 =pd.read_csv('C:/Users/Administrator/Desktop/数据分析课件/PythonFromDAToDS-master/DataSets/bc_data.csv')
df2=df2[["id","diagnosis","area_mean"]]
df2[df2.area_mean> 1000].head()
df2[df2.area_mean> 1000][["id","diagnosis"]].head()

df2.loc[df2.area_mean> 1000,["id","diagnosis"]].head()   #关于上述的表达式，该表达式也是可以的

df4=pd.DataFrame(np.arange(6).reshape(2,3))
df4
df5=pd.DataFrame(np.arange(10).reshape(2,5)) 
df5

df4+df5
df6=df4.add(df5,fill_value=10)  #这个fill_value表示如何将两个表格进行相加，同时将缺失值使用某一个具体值进行替代
df6

s1=pd.Series(np.arange(4))
s1
df6-s1          ##这个命令表示数据框取多少列的信息

df5=pd.DataFrame(np.arange(10).reshape(2,5))
s1=pd.Series(np.arange(3))
df5-s1

df7=pd.DataFrame(np.arange(20).reshape(4,5))
df7+2

df7.cumsum()
df7

df7.rolling(2).sum()   ##这个命令标识按照0轴滚动两两进行求和
df7.rolling(2,axis=1).sum() ##这个命令表示按照1轴两两进行求和

df7.cov()
df7.corr()    ##表示求皮尔逊相关系数  两个变量的变化趋势完全一样，相关系数肯定是等于1

##如果单单只是用类似列表的方式来取值，那么就要使用下面的这种方法，如果说要是使用iloc的方法，那就可以使用简便方法
df2=df2[["id","diagnosis","area_mean"]][2:5]
df2.T



df6
df6>5
df6>s1
df6>(2,18)


df2.describe()  #统计信息
dt = df2[df2.diagnosis=='M']

dt.head()
dt.tail()
df2[df2.diagnosis=='M'].count()
df2[["area_mean","id"]].head()

df2.head(8)
df2.sort_values(by="area_mean",axis=0,ascending=True).head()

df2.sort_index(axis=1).head(3)  ##按照索引排序

df2.sort_index(axis=0,ascending=False).head(3)  ##按照字母顺序来

df2.head(3).to_excel("df3.xls")  ##数据框导出 to_csv

##缺失值数据处理
df2.empty    ##判断数据框当中是否存在缺失值

A=pd.DataFrame(np.array([10,10,20,20]).reshape(2,2),columns=list("ab"),index=list("SW"))
A
list("ab")
B=pd.DataFrame(np.array([1,1,1,2,2,2,3,3,3]).reshape(3,3), columns=list("abc"),index=list("SWT"))
B
C=A+B 
C
A.add(B,fill_value=0) 

A.add(B,fill_value=A.stack().mean())

A.mean()
A.stack() 
A.stack().mean()
C
C.isnull()
C.dropna(axis='index')
C.fillna(0)
C.fillna(method="bfill")   ##这个表示向前传播或者是向后传播，就是拿前一个值插入到缺失值中
C.fillna(method="ffill",axis=1) ##这个表示按列进行前传和后传的操作


##分组操作
df2
df2.groupby("diagnosis")["area_mean"].mean()

df2.groupby("diagnosis")["area_mean"].aggregate(["mean","sum","max","median"])

df2.groupby("diagnosis")["area_mean"].describe()

df2.groupby("diagnosis")["area_mean"].aggregate(["mean","sum"])  ##一个是横向排列，下面的是纵向排列

df2.groupby("diagnosis")["area_mean"].aggregate(["mean","sum"]).unstack()

def myfunc(x):
   x["area_mean"]=x["area_mean"].sum()
   return x

a=df2.groupby("diagnosis").apply(myfunc).head()


import pandas as pd
df = pd.read_excel(r"..\Data\Chapter05.xlsx",sheet_name=2)
df.drop_duplicates() #删除重复的列
df.drop_duplicates(subset = "唯一识别码") #指定判断的列
df.drop_duplicates(subset = ["客户姓名","唯一识别码"])
df.drop_duplicates(subset = ["客户姓名","唯一识别码"],keep = "last") #keep参数（first,last）first表示保留第一个出现的行，last表示保留最后一个出现的行，false表示全部删除

df["唯一识别码"].astype("float64")#将唯一识别码冲int类型转为float类型

##为数据框设置索引
df.columns = ["订单编号","客户姓名","唯一识别码","成交时间"]#header需要设置为None，否则会覆盖第一行数据
df.index = [1,2,3,4,5]

##数据框重新设置索引和列名
df.rename(columns={"订单编号":"新订单编号","客户姓名":"新客户姓名"}) #重命名列索引
df.rename(index = {1:"一",2:"二",3:"三"}) #重命名行索引
df.rename(columns={"订单编号":"新订单编号","客户姓名":"新客户姓名"},index = {1:"一",2:"二",3:"三",4:'四'})#同时重命名列和行索引

##选择某几列
df['客户姓名']
df[['订单编号','客户姓名']]
df.iloc[:,[0,2]]

#请切记，列表中关于取值的操作和pandas当中的iloc列表取值完全是两回事情
#列表中的第一个框其实是规定了行的数量，而第二个框事实上是规定了列的数量


#选择年龄小于200并且唯一识别码小于200，条件用括号括起来
df[(df['年龄']<200) & (df['唯一识别码']<102)]


df = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =0)
#对某一列进行数值替换
df["年龄"].replace(240,33,inplace = True) #第一个值表示表中的值，第二个数字表示要替换的值
df
#对全表中的缺失值进行替换
df.replace(np.NaN,0)

df.replace([240,260,280],35)

df.replace({240:32,260:33,280:34})

df.sort_values(by=["销售ID"])
#按照销售ID进行降序排序
df.sort_values(by=["销售ID"],ascending= False)


df1 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =2)
df1
#默认空值是排在最后面
df1.sort_values( by = ["销售ID"])
#通过设置na_position参数将缺失的值显示在前面，默认参数值是last
df1.sort_values(by = ["销售ID"],na_position = "first")

##按照多列数字进行排序
df3 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =3)
df3
#将需要排序的by里面，然后在设置升降序
df3.sort_values(by=["销售ID","成交时间"],ascending = [True,False])

df4 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =4)
df4

df5 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =1)
df5["销售ID"]
#method取average时与Excel中的RANK.AVG函数一样
df5["销售ID"].rank(method ="average")

df5["销售ID"].rank(method ="first")  ##排列的是index如果两个相同的话取第一个
df5["销售ID"].rank(method ="min")    ##如果两个相同取最小的排名
df5["销售ID"].rank(method ="max")    ##如果两个相同取最大的排名

##删除列方法
df5.drop(["销售ID","成交时间"],axis =1)

df5.drop(df5.columns[[4,5]],axis=1)     ##这两种方法要注意

df5.drop(columns = ["销售ID","成交时间"])
##删除行的方法
df5.drop(["0a","1b"],axis = 0)
df5.drop(df.index[[0,1]])
df5.drop(index = ["0a","1b"])

##删除特定的行
df5[df5["年龄"]<40]

##数值计算
df5["销售ID"].value_counts()
#计算销售ID的值占比
df5["销售ID"].value_counts(normalize = True)

df5["销售ID"].unique()

##采用的是数值查找的形式
df5['年龄'].isin([31,21])

df5.isin(["A2",31])

df6 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =5)
df6
pd.cut(df6["年龄"],bins = [0,3,6,9,10])  ##这个函数是用来进行分组的,十分有用

pd.qcut(df6["年龄"],3)        ##这个函数表示均匀地切分

##插入列数据
df7 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =3)
df7
df7.insert(2,"商品类别",["cat01","cat02","cat03","cat04","cat05"])

df7["商品类别"]= ["cat01","cat02","cat03","cat04","cat05"]
df7

##关于行的插入,没有好的办法，智能将矩阵转秩，然后进行插入
a=df7.T
#再转置则回到原来的结果
df7.T.T
m=df["成交时间"]

df8 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =6)
df8
df8.stack()


###宽表转为长表
import pandas as pd
df = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =7)
df
#设置索引
df.set_index(["Company","Name"])
#将列索引转为行索引
df.set_index(["Company","Name"]).stack()
#重置索引
df.set_index(["Company","Name"]).stack().reset_index()
#重命名索引
df.set_index(["Company","Name"]).stack().reset_index().rename(columns={"level_2":"Year",0:"sale"})

df.melt(id_vars=["Company","Name"],var_name="Year",value_name = "Sale")

##长表转化为宽表
df2 = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =8)
df2

df2.pivot_table(index=["Company","Name"],columns="Year",values="Sale")

###apply函数和applymap函数
df = pd.read_excel(r"C:\Users\Administrator\Desktop\Excel-Python-master\Data\Chapter07.xlsx",sheet_name =9)
df
df["C1"].apply(lambda x:x+1)  ##这个表示局部函数，下面那个表示全局函数

df.applymap(lambda x:x+1)  ##对表内每一个元素都加1

df.apply(lambda x:x+1,axis=0)

df.iloc[0,:]=df.iloc[0,:].apply(lambda x:x+1)  ##apply的按行操作

##算数相加
df = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter08.xlsx",sheet_name = 0)
df
df.index=["S1","S2","S3"]

df["C1"]+df["C2"]

df["C1"]-df["C2"]

df["C1"]*df["C2"]

df["C1"]/df["C2"]

df["C1"]+1
df["C1"]-1

df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter08.xlsx",sheet_name = 0)
#添加行索引
df1.index=["S1","S2","S3"]
df1
df1["C1"] > df1["C2"]
df1["C1"] < df1["C2"]
df1["C1"] != df1["C2"]

df1.count()

df1.count(axis =0)

df.sum(axis=1)

df["C1"].sum()

df.mean()

df.mean( axis =1)

df["C1"].mean()

df.max()

df.max( axis =1)

df["C1"].max()

##求众数
df3 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter08.xlsx",sheet_name=1)
df3.index=["S1","S2","S3"]
df3.mode()

##四分位数
df5 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter08.xlsx",sheet_name=1)
df5.index=["S1","S2","S3","S4","S5"]
df5
df5.quantile(0.25)#求四分之一分数位
df5.quantile(0.75)#求四分之三分数位
df5.quantile(0.25,axis = 1)#求每一行的四分之一分数位

df5.corr()

##关于时间的计算
from datetime import datetime
datetime.now()

datetime.now().year 
datetime.now().month 
datetime.now().day 


datetime.now().weekday()+1      ##周几

datetime.now().isocalendar()    ##年月日的返回值

datetime.now().isocalendar()[1]  ##第几周

##显示日期
datetime.now().date()

datetime.now().time()

datetime.now().strftime("%Y-%m-%d")

index = pd.DatetimeIndex(['2018-01-01','2018-01-02','2018-01-03','2018-01-04','2018-01-05',
                          '2018-01-06','2018-01-07','2018-01-08','2018-01-09','2018-01-10'])
data = pd.DataFrame(np.arange(1,11),columns =["num"],index = index)
data

data["2018"]

data["2018-01-01":"2018-01-05"]

##时间的比较与计算
import pandas as pd
from datetime import datetime
df9 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter06.xlsx",sheet_name = 4)
df9[df9["成交时间"]>datetime(2018,8,8)]

df9[df9["成交时间"] == datetime(2018,8,8)]

df9[(df9["成交时间"]>datetime(2018,8,8))&(df9["成交时间"]< datetime(2018,8,11))]

cha = datetime(2018,5,21,19,50)-datetime(2018,5,18,20,32)
cha

cha.days

cha.seconds

from datetime import timedelta,datetime
date = datetime.now()
date

date+timedelta(days =1)

date+timedelta(seconds = 60)

date - timedelta(days =1)

from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
date = datetime.now()
date
date+Day(1)
date+Hour(1)
date+Minute(10)
date+MonthEnd(1)

###数据分组***
import pandas as pd
df3 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter10.xlsx",sheet_name =0)
df3

df3.groupby("客户分类").count()
g=df3.groupby(df3["客户分类"])
g.mean()
g.describe()

df3.groupby("客户分类").sum()

df3.groupby(["客户分类","区域"]).count()

df3.groupby(["客户分类","区域"]).count().reset_index()

df3.groupby(["客户分类","区域"]).describe()

df3["客户分类"]
df3.groupby(df3["客户分类"]).count()
df3.groupby([df3["客户分类"],df3["用户ID"]]).sum()

df3.groupby(df3["客户分类"])["用户ID"].count()

df3.groupby("客户分类").aggregate(["count","sum"])
df3.groupby("客户分类").aggregate({"用户ID":"count","7月销量":"sum","8月销量":"sum"})

df3.groupby("客户分类").sum()

df3.groupby("客户分类").sum().reset_index()

##数据透析****
df7 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter10.xlsx",sheet_name =0)
pd.pivot_table(df7,values = "用户ID",columns ="区域",index="客户分类",aggfunc="count")

pd.pivot_table(df7,values = "用户ID",columns ="区域",index="客户分类",aggfunc="count",margins = True)

pd.pivot_table(df7,values = "用户ID",columns ="区域",index="客户分类",aggfunc="count",margins = True,fill_value =0)

###这里面主要是有三个参数第一个是values、第二个是columns、第三个是index，主要是这三个参数搞定了的话数据透析也就搞定了
pd.pivot_table(df7,values = ["用户ID","7月销量"],columns="区域",index="客户分类",aggfunc={"用户ID":"count","7月销量":"sum"})

pd.pivot_table(df7,values="用户ID",columns="区域",index="客户分类",aggfunc="count")

pd.pivot_table(df7,values="用户ID",columns="区域",index="客户分类",aggfunc="count").reset_index()


#关于多表拼接
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =0)
df1

df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =1)
df2
###一对一
pd.merge(df1,df2)  ##如果id有两个相似值的话会出现排列组合的情况

df3 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =2)
df3

##多对一
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =2)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =3)
pd.merge(df1,df2,on = "学号")

##多对多
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =4)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =3)
pd.merge(df1,df2)

##指定连接键 on
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =0)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =1)
df1
df2
pd.merge(df1,df2)
##使用on作为键名连接的方式
pd.merge(df1,df2,on="学号")

###指定多个键名
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =0)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =5)
pd.merge(df1,df2,on=["姓名","学号"])    ###两个变量叠加形成一个新的独一无二的索引

df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =6)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =1)
pd.merge(df1,df2,left_on = "编号",right_on = "学号")

df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =7)
df1.set_index("编号")

df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =1)
df2.set_index("学号")

##左右表的连接键均为索引
pd.merge(df1.set_index("编号"),df2.set_index("学号"),left_index = True,right_index = True)

df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/chapter11.xlsx",sheet_name =1)
df2

pd.merge(df1.set_index("编号"),df2,left_index = True,right_on = "学号")

##连接方式，使用how来指定具体的连接方式
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =0)
df1
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =8)
df2
pd.merge(df1,df2,on="学号",how="inner")  ##这个连接方式表示取交集

pd.merge(df1,df2,on="学号",how="left") ##这个连接方式表示取左集合全部

pd.merge(df1,df2,on="学号",how="right")##这个连接方式表示取右集合的全部

pd.merge(df1,df2,on="学号",how="outer") ##外连接，两个表的并集

###表的纵向合并
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =9)
df1
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =10)
df2
pd.concat([df1,df2])

##索引设置
pd.concat([df1.set_index("编号"),df2.set_index("编号")],ignore_index = True)

##重叠数据合并
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =11)
df1
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter11.xlsx",sheet_name =10)
df2
pd.concat([df1.set_index("编号"),df2.set_index("编号")],ignore_index = True)

##删除重复值
pd.concat([df1.set_index("编号"),df2.set_index("编号")],ignore_index = True).drop_duplicates()

##导出为xlsx文件
import pandas as pd
df = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =0 )
df.to_excel(excel_writer = r"C:\Users\Administrator\Desktop\Excel-Python-master\Note\测试文档01.xlsx")

df.to_excel(excel_writer = r"C:\Users\Administrator\Desktop\Excel-Python-master\Note\测试文档02.xlsx",
            sheet_name ="测试")
df.to_excel(excel_writer = r"C:\Users\Administrator\Desktop\Excel-Python-master\Note\测试文档03.xlsx",
            index = False)

df = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =0 )
df.to_excel(excel_writer = r"C:\Users\Administrator\Desktop\Excel-Python-master\Note\测试文档04.xlsx",
            sheet_name = "测试文档",
            index=False,columns = ["用户ID","7月销量","8月销量","9月销量"])

##关于Excel的输出，需要具备以下几个参数：excel_writer、sheet_name、index、columns
##encoding、na_rep=0、

##导出为csv格式文件
df = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =2)
df.to_csv(path_or_buf = r"C:/Users/Administrator/Desktop/Excel-Python-master/Note/测试文档06.csv" ,
          index= False,
          columns = ["用户ID","7月销量","8月销量","9月销量"],
          sep=",",
          na_rep = 0,
          encoding = "gbk" #设置为gbk或者utf-8-sig
         )

###关于多表的输出的问题
df1 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =0)
df2 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =1)
df3 = pd.read_excel(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/Chapter12.xlsx",sheet_name =2)
#声明一个对象
writer = pd.ExcelWriter(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/test03.xlsx",
                        engine = "xlsxwriter")
#将df1、df2、df3写入Excel中的sheet1、sheet2、sheet3
#重命名表1、表2、表3
df1.to_excel(writer,sheet_name ="表1",index=False)
df2.to_excel(writer,sheet_name ="表2",index=False)
df3.to_excel(writer,sheet_name ="表3",index=False)
#保存读写的内容
writer.save()





###关于Python画图的问题
##建立画布
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'  ##字体simhei表示中文中的黑体
#解决负号无法正常显示问题
plt.rcParams["axes.unicode_minus"]= False
#设置为矢量图
%config InlineBackend.figure_format = 'svg'
#建立画布
fig = plt.figure()
#设置画布的高与长
plt.figure(figsize = (8,6))

##用add_subplot函数建立坐标系
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
#建立4个坐标系
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

##用plt.subplot2grid函数建立坐标系
plt.subplot2grid((2,2),(0,0))

import numpy as np
x = np.arange(6)
y = np.arange(6)

plt.subplot2grid((2,2),(0,0))
plt.plot(x,y)
plt.subplot2grid((2,2),(0,1))
plt.bar(x,y)

##用plt.subplot函数建立坐标系
#将图表分成2行2列，并在第1个坐标系里面绘图
plt.subplot(2,2,1)
import numpy as np
x = np.arange(6)
y = np.arange(6)
#在第1个坐标系上做折线图
plt.subplot(2,2,1)
plt.plot(x,y)
#在第4个坐标系上做柱状图
plt.subplot(2,2,4)
plt.bar(x,y)

##用plt.subpllots函数建立坐标系
#将图表整个区域分成2行2列，并将4个坐标系全部返回
fig,axes = plt.subplots(2,2)
import numpy as np
x = np.arange(6)
y = np.arange(6)
#在[0,0]坐标系中绘制折线图
axes[0,0].plot(x,y)
#在[1,1]坐标系中绘制柱状图
axes[1,1].bar(x,y)

###设置坐标轴
plt.subplot(1,1,1)
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlabel("月份")
plt.ylabel("注册量")

##通过设置label参数设置坐标到x和y轴的距离
plt.subplot(1,1,1)
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlabel("月份",labelpad = 10)
plt.ylabel("注册量",labelpad = 10)

#设置坐标轴的样式,坐标轴的样式可以进行调整的
plt.subplot(1,1,1)
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlabel("月份",fontsize="xx-large",color="#70AD47",fontweight="bold")
plt.ylabel("注册量",labelpad = 10)

##设置坐标轴的刻度，这个是设置坐标轴刻度的设置
##plt库中使用xticks、yticks，支持文本相关性质设置，使用方法与xlabel、ylabel的文本相关性质设置方法一致
plt.subplot(1,1,1)
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlabel("月份")
plt.ylabel("注册量")
plt.xticks(np.arange(9),["1月份","2月份","3月份","4月份","5月份","6月份","7月份","8月份","9月份"])
plt.yticks(np.arange(1000,7000,1000),["1000人","2000人","3000人","4000人","5000人","6000人"])

##隐藏坐标轴的刻度
plt.subplot(1,1,1)
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlabel("月份")
plt.ylabel("注册量")
plt.xticks([])
plt.yticks([])

'''
tick_params函数可以对刻度进行设置
axis：对那个轴的刻度线进行设置,x、y、both三个可选
reset：是否重置所有设置，True/False
which：对那种刻度进行设置,major(主刻度线)、minior(次刻度线)、both三个可选
direction：刻度的朝向，in(朝里)、out(朝外)、inout(里外均有)三个可选
length：刻度线长度
width：刻度线的宽度
color：刻度线的颜色
pad：刻度线与刻度标签之间的距离
labelsize：刻度标签大小 labelcolor：刻度标签的颜色
top、bottom、left、right：True/False可选，控制上、下、左、右刻度线是否显示
labeltop、labelbottom、labelleft、labelright：True/False可选，控制上、下、左、右刻度标签是否显示
'''

plt.figure(figsize = (6,8))
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
#在2X1坐标系上的第一个坐标系中绘图
plt.subplot(2,1,1)
plt.plot(x,y)
plt.xlabel("月份")
plt.ylabel("注册人数")
plt.yticks(np.arange(1000,7000,1000),["1000","2000","3000","4000","5000","6000"])
#轴刻度线设置双向且下刻度线不显示
plt.tick_params(axis= "both",which = "both", direction = "in" ,bottom=True)

#在2X1坐标系上的第二个坐标系中绘图
plt.subplot(2,1,2)
plt.plot(x,y)
plt.xlabel("月份")
plt.ylabel("注册人数")
plt.yticks(np.arange(1000,7000,1000),["1000","2000","3000","4000","5000","6000"])
#轴刻度线设置双向且下刻度标签不显示
plt.tick_params(axis= "both",which = "both", direction = "out" ,labelbottom=False)

##设置坐标轴的范围
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.xlim(0,10)
plt.ylim(0,8000)

##坐标轴的轴显示设置
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.axis("off")

'''
网络线设置
通过设置b的值，True来启用网格线
通过axis的值(x、y)控制打开那个轴的网格线
linestyle设置网格线样式
linewidth设置网格线宽度
'''

#设置网格线
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.grid(b= True)

#只启用x轴
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.grid(b= True,axis ="x")

#只启用y轴
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.grid(b= True,axis ="y")

#启用网格线，虚线样式，线宽为1
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.grid(b= True,linestyle="dashed",linewidth =1)

##设置图例
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y,label ="折线图")
plt.bar(x,y,label="柱形图")
plt.legend()

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y,label ="折线图")
plt.bar(x,y,label="柱形图")
plt.legend(loc ="upper left")
plt.legend(loc=2)

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y,label ="折线图")
plt.bar(x,y,label="柱形图")
plt.legend(loc ="upper right",fontsize=9,title="测试")

###图标标题设置
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.title('1-9月份公司注册用户数',loc ="left")

'''
设置数据标签
数据标签实现就是根据实际坐标值在对应的位置显示相应的数值，用text函数实现
plt.text(x,y,str,ha,va,fontsize)

参数	说明
参数(x、y)	分别表示可以在哪里显示数据
str	表示显示的具体数值
horizontalalignment	简称ha,表示str在水平方向的位置，有center、left、right三个值可选
verticalalignment	简称va,表示str在垂直方向的位置，有center、top、bottom三个值可选
fontsize	设置str字体大小
'''

#在（5，1605）处设置y的值
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.title("1-9月份公司注册用户数",loc = "center")
#plt.text(5,1605,"极值点")
plt.text(5,1605,"1605")

#在（5，1605）处设置y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.title("1-9月份公司注册用户数",loc = "center")
for a,b in zip(x,y):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)
    
'''
图表注释
图表注释是为了便于更快的获取图表的信息实现方法如下:
plt.annotate(s,xy,xytext,arrowprops)

参数	说明
s	表示要注释的文本内容
xy	表示要注释的位置
xytext	表示要注释的文本的显示位置
arrowprops	设置箭相关参数、颜色、箭类型设置
'''

x = np.array(["1月份","2月份","3月份","4月份","5月份","6月份","7月份","8月份","9月份"])
y = np.array([886,2335,5710,6482,6120,1605,3813,4428,4631])
plt.plot(x,y)
plt.title("1-9月份公司注册用户数",loc = "center")
plt.annotate("服务器宕机了",
            xy=(5,1605),xytext=(6,1605),
            arrowprops=dict(facecolor="black",arrowstyle="->"))

'''
数据表
数据表就是在图表的基础上在添加一个表格，使用plt库中的table函数
table(cellText=None,cellColours=None,
cellLoc="right",cellWidths=None,
rowLabels=None,rowColours=None,rowLoc="left",
collLabels=None, colColours=None, colLoc="center",
loc="bottom")
table函数中参数说明：
参数	说明
cellText	数据表内的值
cellColours	数据表的颜色
cellLoc	数据表中数值的位置，可选left、right、center
cellWidths	列宽
rowLabels	行标签
rowColours	行标签颜色
rowLoc	行标签位置
colLabels	列标签
colColours	列标签颜色
colLoc	列标签位置
loc	整个数据表的位置，可选坐标系上、下、左、右
'''

plt.subplot(1,1,1)
x= np.array(["东区","南区","西区","北区"])
y1 = np.array([8566,5335,7310,6482])
y2=  np.array([4283,2667,3655,3241])
plt.bar(x,y1,width=0.3,label="任务量")
plt.bar(x,y2,width=0.3,label="完成量")
plt.xticks([])
plt.legend(loc ="upper center",fontsize=9,ncol =2)
plt.title("全国各分区任务量和完成量")
cellText=[y1,y2]
rows = ["任务量","完成量"]
plt.table(cellText= cellText,
         cellLoc="center",
         rowLabels= rows,
         rowColours=["red","yellow"],
         rowLoc="center",
         colLabels= x,
         colColours=["red","yellow","red","yellow"],
         colLoc="left",
         loc="bottom")
plt.savefig(r"C:\Users\Administrator\Desktop\test.jpg")

###绘制常用的图表系列
'''
使用plt库的plot方法，具体参数如下：
plt.plot(x,y,color,linestyle,linewidth,marker,markeredgecolor,markedgwidth,markfacecolor,marksize,label)
x、y分别表示x轴和y轴的数据（必须项）
color表示折线的颜色
linestyle表示线的风格
linewidth表示线的宽度，传入一个表示宽度的浮点数即可
marker表示折线图中每点的标记物的形状
'''
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'
import numpy as np
#建立坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])

#绘图
plt.plot(x,y,color="k",linestyle="dashdot",linewidth=1,marker="o",markersize=5,label="注册用户")

#设置标题及标题位置
plt.title("1-9月份注册用户量",loc="center")

#添加数据标签
for a,b in zip(x,y):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=10)

#设置网格线
plt.grid(True)

#设置图例
plt.legend()

#保存到本地
plt.savefig(r"C:\Users\Administrator\Desktop\plot.png")

'''
绘制柱状图
使用plt库中的bar方法，具体参数如下：
plt.bar(x,height,width=0.8 ,bottom=None,align ="center",color,edgecolor)
x表示在什么位置显示柱状图
height表示每根柱子的高度
width表示每个柱子的宽度，每个柱子的宽度可以都一样，也可以各不相同
bottom表示每个柱子的底部位置，梅根柱子的底部位置可以都不一样，也可以各不相同
align表示柱子的位置与x值的关系，有center、edge两个参数可选，center表示柱子位于x值的中心位置，edge表示柱子位于x值的边缘位置
color表示柱子颜色
edgecolor表示柱子边缘的颜色
'''
#建立坐标系
plt.subplot(1,1,1)

#指明x和y的值
x= np.array(["东区","北区","南区","西区"])
y = np.array([8566,6482,5335,7310,])

#绘图
plt.bar(x,y,width=0.5,align="center",label="任务量")

#设置标题
plt.title("全国分区任务量",loc="center")

#添加数据标签
for a,b in zip(x,y):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=12)

#设置x和y轴的名称
plt.xlabel("分区")
plt.ylabel("任务量")

#显示图例
plt.legend()

#保存到本地
plt.savefig(r"C:\Users\Administrator\Desktop\bar.png")

###柱状对比图

#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4])
y1 = np.array([8566,5335,7310,6482])
y2 = np.array([4283,2667,3655,3241])

#绘图,width=0.3设置柱形图的宽度为0.3
plt.bar(x,y1,width=0.3,label="任务量")
#x+0.3相当于把完成量的每个柱子右移0.3        #对比柱形图十分简单
plt.bar(x+0.3,y2,width=0.3,label="完成量")

#设置标题
plt.title("全国各分区任务量和完成量",loc="center")

#添加标签数据
for a,b in zip(x,y1):
    plt.text(a,b,b,ha="center",va="bottom",fontsize =12)

for a,b in zip(x+0.3,y2):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=12)

#设置x和y轴的名称
plt.xlabel("区域")
plt.ylabel("任务情况")

#设置x轴的刻度
plt.xticks(x+0.15,["东区","南区","西区","北区"])

#设置网格线
plt.grid(False)

#图例设置
plt.legend()

#b保存图片
plt.savefig("C:/Users/Administrator/Desktop/bars.png")


##堆积柱状图
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'
import numpy as np

#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array(["东区","南区","西区","北区"])
y1 = np.array([8566,5335,7310,6482])
y2 = np.array([4283,3241,2667,3655])

#绘图,width=0.3设置柱形图的宽度为0.3
plt.bar(x,y1,width=0.3,label="任务量")
#x+0.3相当于把完成量的每个柱子右移0.3
plt.bar(x,y2,width=0.3,label="完成量")

#设置标题
plt.title("全国各分区任务量和完成量",loc="center")

#添加标签数据
for a,b in zip(x,y1):
    plt.text(a,b,b,ha="center",va="bottom",fontsize =12)

for a,b in zip(x,y2):
    plt.text(a,b,b,ha="center",va="top",fontsize=12)

#设置x和y轴的名称
plt.xlabel("区域")
plt.ylabel("任务情况")

#设置网格线
plt.grid(False)

#图例设置
plt.legend(loc="upper center",ncol=2)

#b保存图片
plt.savefig(r"C:/Users/Administrator/Desktop/bars.png")

'''
绘制条形图
使用plt库中的barh方法参数如下:
plt.barh(y,width,height,align,color,edgecolor)
y表示在什么地方限制柱子,即纵坐标
width表示柱子在横向的宽度,即横坐标 height表示柱子在纵坐标向的高度,即柱子的实际宽度
align表示柱子的对齐方式
color表示柱子的颜色
edgecolor表示柱子边缘的颜色
'''
#建立坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array(["东区","南区","西区","北区"])
y = np.array([8566,5335,7310,6482])

#绘图
plt.barh(x,height=0.5,width=y,align="center")

#设置标题
plt.title("全国各分区任务量",loc="center")

#添加数据标签
for a,b in zip(x,y):
    plt.text(b,a,b, ha="center",va="center",fontsize=12)

#设置x和y轴的名称
plt.xlabel("任务量")
plt.ylabel("区域")

#设置网格线
plt.grid(False)

'''
绘制散点图
使用plt库中的scatter方法参数如下:
plt.scatter(x,y,s,c,marker,linewidths,edgecolors)
x,y 表示散点的位置
s 表示每个点的面积,即散点的大小.如果是一个具体的值时,则是由的点大小都一样.也可以呈现多个值,让每个点的大小都不一样,这时候就成了气泡图了.
c 表示每个点的颜色,如果做只有一种颜色时,则所有的点颜色相同,也可以呈现多哦颜色值,让不同的颜色不同
marker 表示每个点的标记和折线图的中的marker一致
linewidths 表四每个散点的宽度
edgecolors 表示每个散点轮廓的颜色
'''
#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = [5.5, 6.6, 8.1, 15.8, 19.5, 22.4, 28.3, 28.9]
y = [2.38, 3.85, 4.41, 5.67, 5.44, 6.03, 8.15, 6.87]

#绘图
plt.scatter(x,y,marker="o",s=100)

#设置标题
plt.title("1-8月份平均气温与啤酒销量关系图",loc = "center")

#设置x和y轴名称
plt.xlabel("平均气温")
plt.ylabel("啤酒销量")

#设置网格线
plt.grid(False)

#保存到本地
plt.savefig("C:/Users/Administrator/Desktop/scatter01.jpg")​
#设置网格线
plt.grid(False)
​
#保存到本地
plt.savefig("C:/Users/Administrator/Desktop/scatter01.jpg")

#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([5.5, 6.6, 8.1, 15.8, 19.5, 22.4, 28.3, 28.9])
y = np.array([2.38, 3.85, 4.41, 5.67, 5.44, 6.03, 8.15, 6.87])

#绘图
colors = y*10
area = y*100
plt.scatter(x,y, c = colors,marker="o",s=area)

#设置标题
plt.title("1-8月份平均气温与啤酒销量关系图",loc = "center")

#设置x和y轴名称
plt.xlabel("平均气温")
plt.ylabel("啤酒销量")

#添加数据标签
for a,b in zip(x,y):
    plt.text(a,b,b,ha="center",va="center",fontsize=10,color="white")
#设置网格线
plt.grid(False)

#保存到本地
plt.savefig("C:/Users/Administrator/Desktop/scatter02.jpg")
'''
绘制面积图
使用plt库中的stackplot方法参数如下: plt.stackplot(x,y,labels,color)
x,y 表示x和y坐标数值
labels 不同系列图标的图例名
color 不同系列图标的颜色
'''
#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])

#绘图
labels =["注册人数","激活人数"] #指明系列标签
plt.stackplot(x,y1,y2,labels = labels)

#设置标题
plt.title("XXX公司1-9月注册与激活人数",loc ="center")

#设置x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册与激活人数")

#设置网格
plt.grid(False)

#设置图例
plt.legend()
'''
绘制面积图
使用plt库中的stackplot方法参数如下: plt.stackplot(x,y,labels,color)
x,y 表示x和y坐标数值
labels 不同系列图标的图例名
color 不同系列图标的颜色
'''
#建立一个坐标系
plt.subplot(1,1,1)
​
#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
​
#绘图
labels =["注册人数","激活人数"] #指明系列标签
plt.stackplot(x,y1,y2,labels = labels)
​
#设置标题
plt.title("XXX公司1-9月注册与激活人数",loc ="center")
​
#设置x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册与激活人数")
​
#设置网格
plt.grid(False)
​
#设置图例
plt.legend()
​
#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/stackplot.jpg")
'''
绘制树地图
树地图常用来表示同一级中不同列别的占比关系,使用squarify库,具体参数如下:
squarify.plot(size,label,color,value,edgecolor,linewidth)
size 待绘图数据
label 不同列别的图例标签
color 不同列别的颜色
value 不同列别的数据标签
edgecolor 不同列别之间边框的颜色
linewidth 边框线宽
'''
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'

import squarify
import numpy as np
#指定每一块的大小
size = np.array([3.4,0.693,0.585,0.570,0.562,0.531,0.530,0.524,0.501,0.478,0.468,0.436])

#指定每一块标签文字
xingzuo = np.array(["未知","摩羯座","天秤座","双鱼座","天蝎座","金牛座","处女座","双子座","射手座","狮子座","水瓶座","白羊座"])
#指定每一块数值标签
rate = np.array(["34%","6.93%","5.85%","5.70%","5.62%","5.31%","5.30%","5.24%","5.01%","4.78%","4.68%","4.36%"])

#指定每一块的颜色
colors = ["steelblue","#9999ff","red","indianred","green","yellow","orange"]

#绘图
plot = squarify.plot(sizes= size,label= xingzuo,color = colors,value = rate, edgecolor = 'white',linewidth =3)

#设置标题
plt.title("菊粉星座分布",fontdict={'fontsize':12})

#去除坐标轴
plt.axis('off')

#去除上边框和右边框的刻度
plt.tick_params(top=False,right=False)

'''
绘制雷达
雷达图使用的是plt库中的polar方法,polar是用来建立极坐标系的,其实雷达图就是先将各点展示在极坐标系中,然后用线将各点连接起来,具体参数如下:
plt.polar(theta,r,color,marker,linewidth)
theta 每一点在极坐标系中的角度
r 每一点在极坐标系中的半径
color 连接各点之间线的颜色
marker 每点的标记物
linewidth 连接线的宽度
'''
#建立坐标系
plt.subplot(111,polar = True) #参数polar等于True表示建立一个极坐标系
​
dataLenth = 5 #把圆分成5份
#np.linspace表示在指定的间隔内返回均匀间隔的数字
angles = np.linspace(0,2*np.pi,dataLenth,endpoint = False)
labels = ["沟通能力","业务理解能力","逻辑思维能力","快速学习能力","工具使用能力"]
data = [2,3.5,4,4.5,5]
data = np.concatenate((data,[data[0]]))#闭合
angles = np.concatenate((angles,[angles[0]])) #闭合
​
#绘图
plt.polar(angles,data,color='r',marker="o")
​
#设置x轴宽度
plt.xticks(angles,labels)
​
#设置标题
plt.title("某数据分析师纵隔评级")
​
#保存本地
plt.savefig("C:/Users/Administrator/Desktop/polarplot.jpg")
'''
绘制箱形图
箱形图用来反映一组数据的离散情况,使用plt库中的boxplot方法具体参数如下:
plt.boxplot(x,vert,widths,labels)
x 待绘图源数据
vert 箱形图方向,如果为True则表示纵向；如果是False则表示横向，默认为True
widths 箱形图的宽度
labels 箱形图的标签
'''
import numpy as np
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'
#解决负号无法正常显示问题
plt.rcParams["axes.unicode_minus"]= False
#设置为矢量图
%config InlineBackend.figure_format = 'svg'
​
#建立一个坐标系
plt.subplot(1,1,1)
​
#指明X值
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
x = [y1,y2]
​
#绘图
labels = ["注册人数","激活人数"]
plt.boxplot(x,labels = labels,vert = True,widths = [0.2,0.5])
​
#设置标题
plt.title("XXX公司1-9月份注册于激活人数",loc = "center")
​
#设置网格线
plt.grid(False)
​
#保存到本地
plt.savefig(r"C:\Users\Administrator\Desktop\boxplot.jpg")

'''
绘制饼图
饼图也常用来表示一等级中不同类别的占比情况，使用的方法是plt库中的pie方法具体参数如下:
plt.pie(x,explode,labels,colors,autopct,pctdistance,shadow,labeldistance,startangle,radius,counterclock,wedgeprops,textprops,center,frame)
x 待绘图的数据
explode 饼图找哦就能够每一块离心圆的距离
labels 饼图中每一块的标签 color 饼图中每一块的颜色
autopct 控制饼图内数值百分比的格式
pactdistanc 数据标签距中心的距离
shadow 饼图是否有阴影
labeldistance 每一块饼图距离中心的距离 startangle 饼图初始角度
radius 饼图的半径 counterclock 是否让饼图逆时针显示
wedgeprops 饼图内外边缘属性
textprops 饼图中文本相关属性
center 饼图中心位置
frame 是否显示饼图背后的图框
'''
#建立坐标系
plt.subplot(1,1,1)
​
#指明x值
x = np.array([8566,5335,7310,6482])
​
#绘图
labels = ["东区","北区","南区","西区"]
#让第一块离圆心远点
explode = [0.05,0,0,0]
labeldistance = 1.1
plt.pie(x,labels=labels ,autopct='%.0f%%',shadow = True,explode = explode,radius = 1.0 ,labeldistance = labeldistance)
​
#设置标题
plt.title("全国各区域人数占比",loc="center")
​
#保存图表到本地
plt.savefig(r"C:\Users\Administrator\Desktop\pie.jpg")

#建立坐标系
plt.subplot(1,1,1)

#指明x值
x1 = np.array([8566,5335,7310,6482])
x2=  np.array([4283,3241,2667,3655])

#绘图
labels = ["东区","北区","南区","西区"]

plt.pie(x1,labels=labels,radius = 1.0 ,wedgeprops=dict(width=0.3,edgecolor="w"))
plt.pie(x2,radius = 0.7 ,wedgeprops=dict(width=0.3,edgecolor="w"))

#添加注释
plt.annotate("完成量",xy=(0.35,0.35),xytext =(0.7,0.45),arrowprops=dict(facecolor="black",arrowstyle="->"))
plt.annotate("任务量",xy=(0.75,0.20),xytext =(1.1,0.2),arrowprops=dict(facecolor="black",arrowstyle="->"))
#设置标题
plt.title("全国各区域人数占比",loc="center")

#保存图表到本地
plt.savefig(r"C:\Users\Administrator\Desktop\pie01.jpg")
'''
绘制热力图
热力图是将某一事物的响应度反映在图表上，可以快速发现需要重点关注的区域，适应plt库中的imshow方法,具体参数如下:
plt.imshow(x,cmap)
x 表示待绘图的数据，需要矩阵形式
cmap 配色方案，用来避阿明图表渐变的主题色
cmap的所有可选值都是封装在plt.cm里面
'''
import itertools
#几个相关指标之间的相关性
cm = np.array([[1,0.082,0.031,-0.0086],
              [0.082,1,-0.063,0.062],
              [0.031,-0.09,1,0.026],
              [-0.0086,0.062,0.026,1]])
cmap = plt.cm.cool #设置配色方案
plt.imshow(cm,cmap = cmap)
plt.colorbar()#显示右边颜色条
​
#设置x和y周的刻度标签
classes = ["负债率","信贷数量","年龄","家庭数量"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks,classes)
plt.yticks(tick_marks,classes)
​
#将数值像是在指定位置
for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,cm[i,j],horizontalalignment="center")
​
plt.grid(False)  
​
#保存图表到本地
plt.savefig(r"C:\Users\Administrator\Desktop\imshow.jpg")

'''
绘制水平线和垂直线
水平线和垂直线主要用啦做对比参考，使用plt库中的axhline和axvline方法具体参数如下:
plt.axhline(y,xmin,xmax)
plt.axvline(x,ymin,ymax)
x/y 画水平/垂直线上和的横纵坐标 xmin/xmax 水平线起点和终点
ymin/ymax 垂直线起点和终点
'''

#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
​
#建立坐标系
plt.subplot(1,2,1)
​
#绘制一条y等于2且起点是0.2，重点是0.6的水平线
plt.axhline(y=2,xmin=0.2,xmax=0.6)
​
plt.subplot(1,2,2)
​
#绘制一条x等于2且起点是0.2终点是0.6的垂直线
plt.axvline(x=2,ymin=0.2,ymax=0.6)

'''
绘制组合图表
折线图+折线图
'''
#建立一个坐标系
plt.subplot(1,1,1)
​
#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
​
#直接绘制两条折线绘图
plt.plot(x,y1,color="k",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")
plt.plot(x,y2,color="k",linestyle="dashdot",linewidth=1,marker="o",markersize=3,label="激活人数")
​
​
#设置标题
plt.title("XXX公司1-9月注册与激活人数",loc ="center")
​
#添加数据标签
for a,b in zip(x,y1):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)
    
for a,b in zip(x,y2):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)    
​
#设置x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册与激活人数")
​
​
#设置x和y轴的刻度
plt.xticks(np.arange(9),["1月份","2月份","3月份","4月份","5月份","6月份","7月份","8月份","9月份"])
plt.yticks(np.arange(1000,7000,1000),["1000人","2000人","3000人","4000人","5000人","6000人"])
#设置网格
plt.grid(False)
​
#设置图例
plt.legend()
​
#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/plot01.jpg")

#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
plt.plot(x,y1,color="r",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")
plt.bar(x,y2,color="g",label="激活人数")

#设置标题及位置
plt.title("XXX公司1-9月注册与激活人数",loc ="center")

#添加数据标签
for a,b in zip(x,y1):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)
    
for a,b in zip(x,y2):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)    

#设置x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册与激活人数")


#设置x和y轴的刻度
plt.xticks(np.arange(9),["1月份","2月份","3月份","4月份","5月份","6月份","7月份","8月份","9月份"])
plt.yticks(np.arange(1000,7000,1000),["1000人","2000人","3000人","4000人","5000人","6000人"])
#设置网格
plt.grid(False)

#设置图例
plt.legend()

#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/bar02.jpg")
'''
绘制双坐标轴图表
双坐标轴图表就是既有主坐标轴又有次坐标轴图表，两个不同量级的指标放在同一个坐标系中时，就需要开启双坐标轴，比如任务量和我完成率就是连个不同量级的指标

绘制双y轴图表
使用plt库中的twinx方法，绘制流程为：建立坐标系，然后绘制主坐标轴上的图表，再调用plt.twinx方法，最后绘制次坐标轴的图表
'''
#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([0.54459448,0.32392354,0.39002751,
              0.41121879,0.31063077,0.33152276,
              0.92226226,0.02950071,0.15716906])

#绘制主坐标轴上的图表
plt.plot(x,y1,color="g",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")

#设置主x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册量")

#设置主坐标的图例
plt.legend(loc ="upper left")

#调用twinx方法
plt.twinx()

#绘制此坐标轴的图表
plt.plot(x,y2,color="r",linestyle="dashdot",linewidth=1,marker="o",markersize=3,label="激活率")

#设置次x和y轴的名称
plt.xlabel("月份")
plt.ylabel("激活率")

#设置次坐标轴的图例
plt.legend()

#设置标题及位置
plt.title("XXX公司1-9月注册量与激活率",loc ="center")

#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/twinx.jpg")
#建立一个坐标系
plt.subplot(1,1,1)
​
#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([0.54459448,0.32392354,0.39002751,
              0.41121879,0.31063077,0.33152276,
              0.92226226,0.02950071,0.15716906])
​
#绘制主坐标轴上的图表
plt.plot(x,y1,color="g",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")
​
#设置主x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册量")
​
#设置主坐标的图例
plt.legend(loc ="upper left")
​
#调用twinx方法
plt.twinx()
​
#绘制此坐标轴的图表
plt.plot(x,y2,color="r",linestyle="dashdot",linewidth=1,marker="o",markersize=3,label="激活率")
​
#设置次x和y轴的名称
plt.xlabel("月份")
plt.ylabel("激活率")
​
#设置次坐标轴的图例
plt.legend()
​
#设置标题及位置
plt.title("XXX公司1-9月注册量与激活率",loc ="center")
​
#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/twinx.jpg")

'''
绘制双x轴图表
使用plt库中的twiny方法，流程与x双y轴的方法一样。

绘图样式设置
matplotlib库默认样式不是很好看，使用plt.style.available即可查看matplotlib库支持的所有样式
如果需要使用某种样式在程序开头如下代码:plt.style.use(样式名)
'''
plt.style.available
['bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark-palette',
 'seaborn-dark',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'seaborn',
 'Solarize_Light2',
 'tableau-colorblind10',
 '_classic_test']
#建立一个坐标轴
plt.subplot(1,1,1)
#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
plt.plot(x,y1,color="r",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")
plt.bar(x,y2,color="g",label="激活人数")
plt.style.use("ggplot")

#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'

import squarify
import numpy as np
#指定每一块的大小
size = np.array([3.4,0.693,0.585,0.570,0.562,0.531,0.530,0.524,0.501,0.478,0.468,0.436])

#指定每一块标签文字
xingzuo = np.array(["未知","摩羯座","天秤座","双鱼座","天蝎座","金牛座","处女座","双子座","射手座","狮子座","水瓶座","白羊座"])
#指定每一块数值标签
rate = np.array(["34%","6.93%","5.85%","5.70%","5.62%","5.31%","5.30%","5.24%","5.01%","4.78%","4.68%","4.36%"])

#指定每一块的颜色
colors = ["steelblue","#9999ff","red","indianred","green","yellow","orange"]

#绘图
plot = squarify.plot(sizes= size,label= xingzuo,color = colors,value = rate, edgecolor = 'white',linewidth =3)

#设置标题
plt.title("菊粉星座分布",fontdict={'fontsize':12})

#去除坐标轴
plt.axis('off')

#去除上边框和右边框的刻度
plt.tick_params(top=False,right=False)

#b保存到本地
plt.savefig("C:/Users/Administrator/Desktop/squarify.jpg")

#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([0,1,2,3,4,5,6,7,8])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([433,1167,2855,3241,3060,802,1906,2214,2315])
plt.plot(x,y1,color="r",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")
plt.bar(x,y2,color="g",label="激活人数")

#设置标题及位置
plt.title("XXX公司1-9月注册与激活人数",loc ="center")

#添加数据标签
for a,b in zip(x,y1):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)
    
for a,b in zip(x,y2):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=11)    

#设置x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册与激活人数")


#设置x和y轴的刻度
plt.xticks(np.arange(9),["1月份","2月份","3月份","4月份","5月份","6月份","7月份","8月份","9月份"])
plt.yticks(np.arange(1000,7000,1000),["1000人","2000人","3000人","4000人","5000人","6000人"])
#设置网格
plt.grid(False)

#设置图例
plt.legend()

#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/bar02.jpg")


##这个双y轴的图像是excel所不具备的
#建立一个坐标系
plt.subplot(1,1,1)

#指明x和y的值
x = np.array([1,2,3,4,5,6,7,8,9])
y1 = np.array([866,2335,5710,6482,6120,1605,3813,4428,4631])
y2 = np.array([0.54459448,0.32392354,0.39002751,
              0.41121879,0.31063077,0.33152276,
              0.92226226,0.02950071,0.15716906])

#绘制主坐标轴上的图表
plt.plot(x,y1,color="g",linestyle="solid",linewidth=1,marker="o",markersize=3,label="注册人数")

#设置主x和y轴的名称
plt.xlabel("月份")
plt.ylabel("注册量")

#设置主坐标的图例
plt.legend(loc ="upper left")

#调用twinx方法
plt.twinx()

#绘制此坐标轴的图表
plt.plot(x,y2,color="r",linestyle="dashdot",linewidth=1,marker="o",markersize=3,label="激活率")

#设置次x和y轴的名称
plt.xlabel("月份")
plt.ylabel("激活率")

#设置次坐标轴的图例
plt.legend()

#设置标题及位置
plt.title("XXX公司1-9月注册量与激活率",loc ="center")

#保存图片到本地
plt.savefig("C:/Users/Administrator/Desktop/twinx.jpg")


####报表自动化        这一段要重点学习
import pandas as pd
from datetime import datetime
data = pd.read_csv(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/order-14.1.csv", sep=",",engine = "python",encoding="gbk",parse_dates=["成交时间"])
data.head()
data.info()
this_month = data[(data["成交时间"]>= datetime(2018,2,1)) & (data["成交时间"]<= datetime(2018,2,28))]
last_month = data[(data["成交时间"]>= datetime(2018,1,1)) & (data["成交时间"]<= datetime(2018,1,31))]
same_month = data[(data["成交时间"]>= datetime(2017,2,1)) & (data["成交时间"]<= datetime(2017,2,28))]

def get_month_data(data):
    sale = (data["单价"]*data["销量"]).sum()
    traffic = data["订单ID"].drop_duplicates().count()
    s_t = sale/traffic
    return (sale,traffic,s_t)
sale_1,traffic_1,s_t_1 = get_month_data(this_month)
sale_2,traffic_2,s_t_2 = get_month_data(last_month)
sale_3,traffic_3,s_t_3 = get_month_data(same_month)

report = pd.DataFrame([[sale_1,sale_2,sale_3],
                       [traffic_1,traffic_2,traffic_3],
                       [s_t_1,s_t_2,s_t_3]],
                     columns = ["本月累计","上月同期","去年同期"],
                     index =["销售额","客流量","客单价"])
report

#添加同比和环比字段
report["环比"] = report["本月累计"]/report["上月同期"] -1
report["同比"] = report["本月累计"]/report["去年同期"] -1
report
report.to_csv(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/order.csv",encoding = "utf-8-sig")


import pandas as pd
from datetime import datetime
data = pd.read_csv(r"C:/Users/Administrator/Desktop/Excel-Python-master/Data/order-14.3.csv", sep=",",engine = "python",encoding="gbk",parse_dates=["成交时间"])
data.head()
data.groupby("类别ID")["销量"].sum().reset_index().sort_values(by="销量",ascending = False).head(10)
pd.pivot_table(data,index="商品ID",values ="销量",aggfunc = "sum").reset_index().sort_values(by="销量",ascending = False).head(10)
data["销售额"] = data["销量"]*data["单价"]
data.groupby("门店编号")["销售额"].sum()
data.groupby("门店编号")["销售额"].sum()/data["销售额"].sum()

#绘制饼图
#导入matplotlib库中的pyplot并起名为plt
import matplotlib.pyplot as plt
#让画布直接在jupyter Notebook中展示出来
%matplotlib inline
#解决中文乱码问题
plt.rcParams["font.sans-serif"]='SimHei'
#解决负号无法正常显示问题
plt.rcParams["axes.unicode_minus"]= False
#设置为矢量图
%config InlineBackend.figure_format = 'svg'
#建立画布
fig = plt.figure()
(data.groupby("门店编号")["销售额"].sum()/data["销售额"].sum()).plot.pie()

#提取小时数
data["小时"] = data["成交时间"].map(lambda x:int(x.strftime("%H")))
#对小时和订单去重
tracffic = data[["小时","订单ID"]].drop_duplicates()
#求每个小时的客流量
tracffic.groupby("小时")["订单ID"].count()
#绘制折线图
tracffic.groupby("小时")["订单ID"].count().plot()







