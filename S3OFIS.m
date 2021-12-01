function [EstimatedLabels1,timee]=S3OFIS(datatra,labeltra,datates,granlevel,chunksize)
data1=datatra;
label1=labeltra;
lambda0=1;
lambda1=2;
[L1,W]=size(data1);
numbatch1=ceil(L1/chunksize);
%%
unilabel=unique(label1);
numlabel=length(unilabel);
seq11=[1:chunksize:L1,L1];
averdist=0;
Ldata=0;
timee=0;
tic
for ii=1:1:numbatch1
    data10=data1(seq11(ii):1:seq11(ii+1)-1,:);
    label10=label1(seq11(ii):1:seq11(ii+1)-1);
    dist00=pdist(data10,'euclidean').^2;
    for tt=1:granlevel
        dist00(dist00>mean(dist00))=[];
    end
    averdist=(averdist*Ldata+mean(dist00)*length(label10))/(Ldata+length(label10));
    Ldata=Ldata+length(label10);
    if ii==1
        TrainedClassifier=training([],data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
    if ii>1
        TrainedClassifier=training(TrainedClassifier,data10,label10,granlevel,lambda0,lambda1,unilabel);
    end
end
timee=timee+toc;
%%
data0=datates;
[L0,W]=size(data0);
numbatch0=ceil(L0/chunksize);
seq00=[1:chunksize:L0,L0];
tic
for ii=1:1:numbatch0
    tempseq=seq00(ii):1:seq00(ii+1)-1;
    data00=data0(tempseq,:);
    LC=length(tempseq);
    pseduolabel=zeros(LC,1);
    dist00=pdist(data00,'euclidean').^2;
    for tt=1:granlevel
        dist00(dist00>mean(dist00))=[];
    end
    averdist=(averdist*Ldata+mean(dist00)*length(tempseq))/(Ldata+length(tempseq));
    Ldata=Ldata+length(tempseq);
    %%
    [label_est,dist2]=testing(TrainedClassifier,data00,unilabel,numlabel);
    C=exp(-1*(dist2)./(2*averdist));
    C1=[];
    C2=[];
    for tt=1:1:numlabel
        C1=[C1,C(:,:,tt)];
        C2=[C2,ones(1,numlabel)*tt];
    end
    Idx=[];
    for tt=1:1:LC
        [~,seq]=sort(C1(tt,:),'descend');
        C3=C2(seq);
        C3=C3(1:1:numlabel);
        [UD,x1,x2]=unique(C3);
        F = histc(x2,1:numel(x1));
        [x1,x2]=max(F);
        if x1>=ceil((numlabel+0.1)/2)
            Idx=[Idx;tt];
        end
    end
    pseduolabel(Idx)=label_est(Idx);
    pseudolabel10=pseduolabel(Idx);
    data10=data00(Idx,:);
    %%
    if isempty(pseudolabel10)~=1
        [TrainedClassifier]=training(TrainedClassifier,data10,pseudolabel10,granlevel,lambda0,lambda1,unilabel);
    end
end
timee=timee+toc;
data0=datates;
Output0=ones(size(data0,1),numlabel);
[label_est,dist2]=testing(TrainedClassifier,data0,unilabel,1);
[x1,x2,x3]=size(dist2);
dist3=reshape(mean(dist2,2),[x1,x3]);
Output0=Output0.*exp(-1*dist3./(2*averdist));
[~,EstimatedLabels1]=max(Output0,[],2);
end
function [TrainedClassifier]=training(TrainedClassifier,DTra1,LTra1,GranLevel,lambda0,lambda1,seq)
data_train=DTra1;
label_train=LTra1;
N=length(seq);
if isempty(TrainedClassifier)==1
    CN=zeros(N,1);
    averdist=zeros(N,1);
    centre={};
    data_train1={};
    for ii=1:1:N
        centre{ii}=[];
        data_train1{ii}=data_train(label_train==seq(ii),:);
        if isempty(data_train1{ii})~=1
            [CN0,W]=size(data_train1{ii});
            dist00=pdist(data_train1{ii},'euclidean').^2;
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];
            end
            averdist(ii)=mean(dist00);
            if isnan(averdist(ii))==1
                averdist(ii)=0;
            end
            CN(ii)=CN(ii)+CN0;
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
        end
    end
    centre0=centre;
end
if isempty(TrainedClassifier)==0
    centre0=TrainedClassifier.centre;
    averdist=TrainedClassifier.averdist;
    CN=TrainedClassifier.CN;
    centre={};
    data_train1={};
    for ii=1:1:N
        centre{ii}=[];
        data_train1{ii}=data_train(label_train==seq(ii),:);
        if isempty(data_train1{ii})~=1
            [CN0,W]=size(data_train1{ii});
            dist00=pdist(data_train1{ii},'euclidean').^2;
            for tt=1:GranLevel
                dist00(dist00>mean(dist00))=[];
            end
            averdist(ii)=(CN(ii)*averdist(ii)+CN0*mean(dist00))/(CN(ii)+CN0);
            if isnan(averdist(ii))==1
                averdist(ii)=0;
            end
            CN(ii)=CN(ii)+CN0;
            [centre{ii}]=online_training_Euclidean(data_train1{ii},averdist(ii));
        end
    end
    [centre0]=CombiningCentres(centre0,centre,averdist,N,lambda0,lambda1);
end
TrainedClassifier.centre=centre0;
TrainedClassifier.averdist=averdist;
TrainedClassifier.CN=CN;
end
function [centre0]=CombiningCentres(centre0,centre,thresholddistance,N,lambda0,lambda1)
La1=[];
La2=[];
CC1=[];
CC2=[];
for ii=1:1:N
    CC1=[CC1;centre0{ii}];
    CC2=[CC2;centre{ii}];
    La1=[La1;ones(size(centre0{ii},1),1)*ii];
    La2=[La2;ones(size(centre{ii},1),1)*ii];
end
if isempty(CC1)~=1 && isempty(CC2)~=1
    dist11=pdist2(CC1,CC2).^2;
    for ii=1:1:N
        seq11=find(La1==ii);
        seq22=find(La1~=ii);
        seq33=find(La2==ii);
        if isempty(seq11)~=1 && isempty(seq11)~=1
            %%
            dist1=dist11(seq11,seq33);
            seq1=min(dist1,[],1);
            seq2=find(seq1>=thresholddistance(ii)*lambda0);
            dist2=dist11(seq22,seq33);
            dist3=repmat(thresholddistance(La1(seq22))*lambda1,1,length(seq33));
            dist4=dist2-dist3;
            seq44=min(dist4,[],1);
            seq4=find(seq44<=0);
            centre0{ii}=[centre0{ii};centre{ii}(unique([seq2,seq4]),:)];
        else
            centre0{ii}=[centre0{ii};centre{ii}];
        end
    end
elseif isempty(CC1)==1
    centre0=centre;
end
end
function [centre]=online_training_Euclidean(data,averdist)
[L,W]=size(data);
centre=data(1,:);
member=1;
for ii=2:1:L
    [dist3,pos3]=min(pdist2(data(ii,:),centre,'euclidean').^2);
    if dist3>averdist
        centre(end+1,:)=data(ii,:);
        member(end+1,1)=1;
    else
        centre(pos3,:)=(member(pos3,1)*centre(pos3,:)+data(ii,:))/(member(pos3,1)+1);
        member(pos3,1)=member(pos3,1)+1;
    end
end
end
function [label_est,dist2]=testing(TrainedClassifier,data_test,seq,K)
centre=TrainedClassifier.centre;
N=length(centre);
L=size(data_test,1);
dist=zeros(L,N);
dist2=zeros(L,K,N);
for i=1:1:N
    if isempty(centre{i})~=1
        tempseq=pdist2(data_test,centre{i},'euclidean').^2;
        dist(:,i)=min(tempseq,[],2);
        tempseq=sort(tempseq,2,'ascend');
        K1=min([length(centre{i}(:,1)),K]);
        dist2(:,1:1:K1,i)=tempseq(:,1:1:K1);
    end
end
[~,label_est]=min(dist,[],2);
label_est=seq(label_est);
end
