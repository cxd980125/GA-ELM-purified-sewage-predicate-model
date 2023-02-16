function [IW,B,LW,TF,TYPE] = elmtrain(P,T,N,parameter,TF,TYPE)

[R,~] = size(P);
[~,Q] = size(T);

if nargin < 3
    N = size(P,2);
end

if nargin < 5
    TF = 'sig';
end
if nargin < 6
    TYPE = 0;
end

if TYPE  == 1
    T  = ind2vec(T);
end

try
    if length(parameter)==1
        parameter=parameter*ones(R*Q+N,1);
    end
    IW=reshape(parameter(1:R*N),N,R);                 %输入层和隐含层的权值
    B=reshape(parameter(R*N+1:end),N,1);            %隐含层的偏置
catch
    IW = rand(N,R) * 2 - 1;
    B = rand(N,1);
    warning('Problem using function. Assigning default values.');
end


BiasMatrix = repmat(B,1,Q);
% 求解隐含层的输出值
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% 求解输出层的权值，通过求逆的方法，得到LW，得到训练好的模型结构。
LW = pinv(H') * T';
