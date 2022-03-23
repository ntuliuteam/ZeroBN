% Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
%
% This source code is licensed under the Apache-2.0 license found in the
% LICENSE file in the root directory of this source tree.

P = [1 1
1 1
1 1
1 1] %P is the parameters
T = [1 1] %T is the latency label

T=power(T,-(1/8)); % covert into normal distribution

[p1,minp,maxp,t1,mint,maxt]=premnmx(P,T);

net=newff(minmax(P),[16,12,1],{'tansig','tansig','purelin'},'trainlm');

net.trainParam.epochs = 5000;

net.trainParam.goal=0.00001;

[net,tr]=train(net,p1,t1);