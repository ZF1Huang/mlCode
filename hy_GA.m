clear;
clc;
pc=0.9;%交叉率
pm=0.01; %变异率
population_size=40; %种群数量
gene_length=22;%基因个数
fun=@(x)x*sin(10*pi*x)+2;%目标函数
g_space=round(rand(population_size,gene_length));%随机生成初始种群
optimal_x=[];
optimal_value=[];
average_value=[];%初始化相关变量
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
        g_binary=[g_binary;b];%格式转换，这样才能为进制转换提供正确格式
    end
    g_decimal=bin2dec(g_binary)';%进制转换
    g_decimal=g_decimal*3/(2^22-1)-1;%比例缩放
    value=zeros(1,population_size);
    prob=zeros(1,population_size);
    prob_sum=zeros(1,population_size);
    mating_pool=[];
    max_index=1;
    min_index=1; %初始化相关变量
    for i=1:population_size
        value(i)=fun(g_decimal(i));
        if value(i)>value(max_index)
            max_index=i; %寻找最大者
        end
        if value(i)<value(min_index)
            min_index=i; %寻找最小者
        end
    end
    if iter~=1
        average_value=[average_value mean(value)];%保存均值，为画平均适应度曲线
        if value(max_index)>elitist_value
            elitist=g_space(max_index,:);
            elitist_decimal=g_decimal(max_index);
            elitist_value=value(max_index);%设置新的精英个体
        else
        g_space(min_index,:)=elitist;
        g_decimal(min_index)=elitist_decimal;
        value(min_index)=elitist_value;%用精英个体替换原来的最小个体
        end
    elseif iter==1 %迭代次数为1时，只需要计算精英个体
        average_value=[average_value mean(value)];
        elitist=g_space(max_index,:);
        elitist_decimal=g_decimal(max_index);
        elitist_value=value(max_index);
    end
    optimal_x=[optimal_x elitist_decimal];
    optimal_value=[optimal_value elitist_value];%保存最优值，为画最大适应度曲线
    for i=1:population_size
        prob(i)=value(i)/sum(value);%计算概率
    end
    for i=1:population_size %计算累计概率
        if i==1
            prob_sum(i)=prob(i);
        else
            prob_sum(i)=prob(i)+prob_sum(i-1);
        end
    end
    randnum=rand(1,population_size); %取配对池
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
    if rand_cross<pc %交叉
        for i=1:2:population_size-1
            cross_point=unidrnd(gene_length-1); %取交叉点
            for j=cross_point:gene_length
                temp=gene_after_cross(i+1,j); %交换基因
                gene_after_cross(i+1,j)=gene_after_cross(i,j);
                gene_after_cross(i,j)=temp;
            end
        end
    end
    gene_after_mut=gene_after_cross;
    for i=1:population_size %变异
       mut_loc=unidrnd(gene_length); %取变异点
            if rand<pm 
                if gene_after_mut(i,mut_loc)==1 %执行变异操作
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
xlabel('迭代次数')
ylabel('适应度')
legend('最大适应度','平均适应度')
