clear;
clc;
pc=0.9;%������
pm=0.01; %������
population_size=40; %��Ⱥ����
gene_length=22;%�������
fun=@(x)x*sin(10*pi*x)+2;%Ŀ�꺯��
g_space=round(rand(population_size,gene_length));%������ɳ�ʼ��Ⱥ
optimal_x=[];
optimal_value=[];
average_value=[];%��ʼ����ر���
for iter=1:500
    g_binary=[];
    for k=1:population_size
        for i=1:gene_length
            a=g_space(k,:);
            if i==1
                b=num2str(a(i));
            else
                b=[b,num2str(a(i))];
            end
        end
        g_binary=[g_binary;b];%��ʽת������������Ϊ����ת���ṩ��ȷ��ʽ
    end
    g_decimal=bin2dec(g_binary)';%����ת��
    g_decimal=g_decimal*3/(2^22-1)-1;%��������
    value=zeros(1,population_size);
    prob=zeros(1,population_size);
    prob_sum=zeros(1,population_size);
    mating_pool=[];
    max_index=1;
    min_index=1; %��ʼ����ر���
    for i=1:population_size
        value(i)=fun(g_decimal(i));
        if value(i)>value(max_index)
            max_index=i; %Ѱ�������
        end
        if value(i)<value(min_index)
            min_index=i; %Ѱ����С��
        end
    end
    if iter~=1
        average_value=[average_value mean(value)];%�����ֵ��Ϊ��ƽ����Ӧ������
        if value(max_index)>elitist_value
            elitist=g_space(max_index,:);
            elitist_decimal=g_decimal(max_index);
            elitist_value=value(max_index);%�����µľ�Ӣ����
        else
        g_space(min_index,:)=elitist;
        g_decimal(min_index)=elitist_decimal;
        value(min_index)=elitist_value;%�þ�Ӣ�����滻ԭ������С����
        end
    elseif iter==1 %��������Ϊ1ʱ��ֻ��Ҫ���㾫Ӣ����
        average_value=[average_value mean(value)];
        elitist=g_space(max_index,:);
        elitist_decimal=g_decimal(max_index);
        elitist_value=value(max_index);
    end
    optimal_x=[optimal_x elitist_decimal];
    optimal_value=[optimal_value elitist_value];%��������ֵ��Ϊ�������Ӧ������
    for i=1:population_size
        prob(i)=value(i)/sum(value);%�������
    end
    for i=1:population_size %�����ۼƸ���
        if i==1
            prob_sum(i)=prob(i);
        else
            prob_sum(i)=prob(i)+prob_sum(i-1);
        end
    end
    randnum=rand(1,population_size); %ȡ��Գ�
    for i=1:population_size
        if randnum(i)<=prob_sum(1)
            temp=g_space(1,:);
            mating_pool=[mating_pool;temp];
        else
            for j=2:population_size
                if randnum(i)<=prob_sum(j) && randnum(i)>prob_sum(j-1)
                    temp=g_space(j,:);
                    mating_pool=[mating_pool;temp];
                end
            end
        end
    end
    rand_cross=rand;
    gene_after_cross=mating_pool;
    if rand_cross<pc %����
        for i=1:2:population_size-1
            cross_point=unidrnd(gene_length-1); %ȡ�����
            for j=cross_point:gene_length
                temp=gene_after_cross(i+1,j); %��������
                gene_after_cross(i+1,j)=gene_after_cross(i,j);
                gene_after_cross(i,j)=temp;
            end
        end
    end
    gene_after_mut=gene_after_cross;
    for i=1:population_size %����
       mut_loc=unidrnd(gene_length); %ȡ�����
            if rand<pm 
                if gene_after_mut(i,mut_loc)==1 %ִ�б������
                    gene_after_mut(i,mut_loc)=0;
                else
                    gene_after_mut(i,mut_loc)=1;
                end
            end
    end
     g_space=gene_after_mut;   
end
plot(1:iter,optimal_value)
hold on;
plot(1:iter,average_value)
xx=optimal_x(iter)
yy=optimal_value(iter)
xlabel('��������')
ylabel('��Ӧ��')
legend('�����Ӧ��','ƽ����Ӧ��')
