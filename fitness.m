function error = fitness(x)
%�ú�������������Ӧ��ֵ

load para hiddennum_best inputn outputn output_train inputn_test outputps output_test

hiddennum=hiddennum_best;
%ѵ������ѧϰ��ģ��
[IW,B,LW,TF,TYPE] = elmtrain(inputn,outputn,hiddennum,x);  %xΪ�Ż��Ĳ���������elmtrain��������ѵ��ģ��

%��ѵ���õ�ģ�ͽ��з���
an0=elmpredict(inputn,IW,B,LW,TF,TYPE); 
train_simu=mapminmax('reverse',an0,outputps);

%��ѵ���õ�ģ�ͽ���Ԥ��
an=elmpredict(inputn_test,IW,B,LW,TF,TYPE); 
test_simu=mapminmax('reverse',an,outputps);

% error=mse(output_test,test_simu);   %��Ӧ�Ⱥ���ѡȡΪ���Լ��ľ�������Ӧ�Ⱥ���ֵԽС������ģ�͵�Ԥ�⾫��Խ�ߣ�ע��newff�������BP�������˽�����֤�����ѡ���������Ԥ�������Ϊ��Ӧ�Ⱥ����Ǻ���
error=(mse(output_train,train_simu)+mse(output_test,test_simu))/2; %��Ӧ�Ⱥ���ѡȡΪѵ��������Լ�����ľ������ƽ��ֵ����Ӧ�Ⱥ���ֵԽС������ѵ��Խ׼ȷ���Ҽ��ģ�͵�Ԥ�⾫�ȸ��á�



