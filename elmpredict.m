function Y = elmpredict(P,IW,B,LW,TF,TYPE)
% 计算隐含层的输出值
Q = size(P,2);
BiasMatrix = repmat(B,1,Q);
tempH = IW * P + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% 计算输出层的仿真输出值
Y = (H' * LW)';
if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [~,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end
       
    
