%% ��ʼ��
clear
close all
clc
format shortg
warning off
addpath('func_defined')

%% ���ݶ�ȡ
data=xlsread('������.xlsx','Sheet1','A1:N342'); %%ʹ��xlsread������ȡEXCEL�ж�Ӧ��Χ�����ݼ���  
opts.PreserveVariableNames = true;
%�����������
input=data(:,[1,2,3,4,5,7,8,9,10,11]);    %data�ĵ�һ��-�����ڶ���Ϊ����ָ�� input=data(:,1:end-1)
output=data(:,6);  %data�������һ��Ϊ�����ָ��ֵ (:,end)

N = length(output);   %ȫ��������Ŀ
testNum = 15;   %�趨����������Ŀ 15
trainNum = N-testNum;    %����ѵ��������Ŀ

%% ����ѵ���������Լ�
input_train = input(1:trainNum,:)';
output_train = output(1:trainNum)';
input_test = input(trainNum + 1:trainNum+testNum,:)';
output_test = output(trainNum + 1:trainNum+testNum)';

%% ���ݹ�һ��
[inputn,inputps] = mapminmax(input_train,-1,1);
[outputn,outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply',input_test,inputps);

%% ��ȡ�����ڵ㡢�����ڵ����
inputnum = size(input,2);
outputnum = size(output,2);
disp('/////////////////////////////////')
disp('����ѧϰ��ELM�ṹ...')
disp(['�����Ľڵ���Ϊ��',num2str(inputnum)])
disp(['�����Ľڵ���Ϊ��',num2str(outputnum)])
disp(' ')
disp('������ڵ��ȷ������...')

%ȷ��������ڵ����
%ע�⣺BP������ȷ��������ڵ�ķ����ǣ����þ��鹫ʽhiddennum=sqrt(m+n)+a��mΪ�����ڵ������nΪ�����ڵ������aһ��ȡΪ1-10֮�������
%�ڼ���ѧϰ���У��þ��鹫ʽ������ʧЧ�����ýϴ�ķ�Χ����������ڵ���Ŀ��ȷ�����ɡ�

MSE = 1e+5; %��ʼ����С���
for hiddennum = 20:50  % �������� 20~30                    
    
    %��ѵ������ѵ������ѧϰ��ģ��
   [IW0,B0,LW0,TF,TYPE] = elmtrain(inputn,outputn,hiddennum);
   
    %��ѵ��������
    an0 = elmpredict(inputn,IW0,B0,LW0,TF,TYPE);  %������
    mse0 = mse(outputn,an0);  %����ľ������
    disp(['������ڵ���Ϊ',num2str(hiddennum),'ʱ��ѵ�����ľ������Ϊ��',num2str(mse0)])

    %������ѵ�������ڵ�
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['��ѵ�������ڵ���Ϊ��',num2str(hiddennum_best),'����Ӧ�ľ������Ϊ��',num2str(MSE)])

save para hiddennum_best inputn outputn output_train inputn_test outputps output_test

%% ѵ�����������ڵ�ļ���ѧϰ��ģ��
disp(' ')
disp('ELM����ѧϰ����')
[IW0,B0,LW0,TF,TYPE] = elmtrain(inputn,outputn,hiddennum_best);

%% ģ�Ͳ���
an0=elmpredict(inputn_test,IW0,B0,LW0,TF,TYPE); %��ѵ���õ�ģ�ͽ��з���
test_simu0=mapminmax('reverse',an0,outputps); % Ԥ��������һ��
%���ָ��
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);

%% �Ŵ��㷨Ѱ����Ȩֵ��ֵ
disp(' ')
disp('GA�Ż�ELM����ѧϰ����')
%��ʼ��ga����
PopulationSize_Data=30;   %��ʼ��Ⱥ��ģ
MaxGenerations_Data=80;   %����������
CrossoverFraction_Data=0.8;  %�������
MigrationFraction_Data=0.2;   %�������
nvars=inputnum*hiddennum_best+hiddennum_best;    %�Ա�������
%�Ա�������
lb=[-ones(inputnum*hiddennum_best,1)          %����㵽�����������Ȩֵ��Χ��[-1 1]    ����Ϊ-1
    zeros(hiddennum_best,1)];              %��������ֵ��Χ��[0 1]  ����Ϊ0
%�Ա�������
ub=ones(nvars,1);


%�����Ŵ��㷨����
options = optimoptions('ga');
options = optimoptions(options,'PopulationSize', PopulationSize_Data);
options = optimoptions(options,'CrossoverFraction', CrossoverFraction_Data);
options = optimoptions(options,'MigrationFraction', MigrationFraction_Data);
options = optimoptions(options,'MaxGenerations', MaxGenerations_Data);
options = optimoptions(options,'SelectionFcn', @selectionroulette);   %���̶�ѡ��
options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);   %���㽻��
options = optimoptions(options,'MutationFcn', {  @mutationgaussian [] [] });   %��˹����
options = optimoptions(options,'Display', 'off');    %��off��Ϊ����ʾ�������̣���iter��Ϊ��ʾ��������
options = optimoptions(options,'PlotFcn', { @gaplotbestf });    %�����Ӧ����ͼ

%���
[bestx,fval] = ga(@fitness,nvars,[],[],[],[],lb,ub,[],[],options);
delete('para.mat')

%% �Ż���Ĳ���ѵ��ELM����ѧϰ��ģ��
[IW1,B1,LW1,TF,TYPE] = elmtrain(inputn,outputn,hiddennum_best,bestx);          %IW1   B1  LW1Ϊ�Ż����ELM��õ�ѵ������

%% �Ż����ELMģ�Ͳ���
an1=elmpredict(inputn_test,IW1,B1,LW1,TF,TYPE); 
test_simu1=mapminmax('reverse',an1,outputps);

%���ָ��
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);

%% ��ͼ
figure
plot(output_test,'g-.o','linewidth',1)
hold on
plot(test_simu0,'b-*','linewidth',1)
hold on
plot(test_simu1,'r-v','linewidth',1)
legend('��ʵֵ','ELMԤ��ֵ','GA-ELMԤ��ֵ')
xlabel('�����������')
ylabel('ָ��ֵ')
title('�Ż�ǰ���ELMģ��Ԥ��ֵ����ʵֵ�Ա�ͼ')

figure
plot(error0,'b-*','markerfacecolor','r')
hold on
plot(error1,'r-v','markerfacecolor','r')
legend('ELMԤ�����','GA-ELMԤ�����')
xlabel('�����������')
ylabel('Ԥ��ƫ��')
title('�Ż�ǰ���ELMģ��Ԥ��ֵ����ʵֵ���Ա�ͼ')

disp(' ')
disp('/////////////////////////////////')
disp('��ӡ������')
disp('�������     ʵ��ֵ      ELMԤ��ֵ  GA-ELMֵ   ELM���   GA-ELM���')
for i=1:testNum
    disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
end





