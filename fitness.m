function error = fitness(x)
%该函数用来计算适应度值

load para hiddennum_best inputn outputn output_train inputn_test outputps output_test

hiddennum=hiddennum_best;
%训练极限学习机模型
[IW,B,LW,TF,TYPE] = elmtrain(inputn,outputn,hiddennum,x);  %x为优化的参数，赋给elmtrain函数进行训练模型

%用训练好的模型进行仿真
an0=elmpredict(inputn,IW,B,LW,TF,TYPE); 
train_simu=mapminmax('reverse',an0,outputps);

%用训练好的模型进行预测
an=elmpredict(inputn_test,IW,B,LW,TF,TYPE); 
test_simu=mapminmax('reverse',an,outputps);

% error=mse(output_test,test_simu);   %适应度函数选取为测试集的均方误差，适应度函数值越小，表明模型的预测精度越高，注意newff函数搭建的BP，产生了交叉验证，因此选另外的数据预测误差作为适应度函数是合理。
error=(mse(output_train,train_simu)+mse(output_test,test_simu))/2; %适应度函数选取为训练集与测试集整体的均方误差平均值，适应度函数值越小，表明训练越准确，且兼顾模型的预测精度更好。



