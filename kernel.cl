#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void leaveOneOut(
constant float * params0,
global float * matrix, //NumberOfVectors * (NetLength+2),
constant int * params1,
private float * weight, //NumberOfClasses * (NetLength+1)

private int * mixNum,
private float * output,

global int * answer,
global float * outError,
global float * NumberOfErrors,
constant int * randArr)
{

    float ecrit = params0[0];
    float lrate = params0[1];
    float temp = params0[2];
    int NumberOfVectors = params1[0];
    int NumOfClasses = params1[1];
    int NetLength = params1[2];

    float currentError = 2. * ecrit;
    int type=0;
    int randCounter = 12*get_global_id(0);

    outError[get_global_id(0)] = 0.;
    answer[get_global_id(0)] = 1;


    //set zero weights
    //weight[(NetLength+1)*(NumOfClasses-1) + NetLength] = 0.;
    for(int j = 0; j < NumOfClasses; ++j)
    {
        for(int i = 0; i < NetLength+1; ++i)
        {
            weight[j * (NetLength+1) + i] = 0.; //error?crash
        }
    }


    for(int i=0; i<NumberOfVectors; ++i)
    {
        mixNum[i]=i;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    printf("init %d\n", get_global_id(0));
    for(int i=0; i<NumberOfVectors; ++i)
    {
        printf("%d ", mixNum[i]);
    }
    printf("\n\n");

    barrier(CLK_LOCAL_MEM_FENCE);


    //NumberOfErrors = new int[NumOfClasses];
    //helpString="";
    for(int i = 0; i < NumOfClasses; ++i)
    {
        NumberOfErrors[i] = 0;
    }


    int a1, a2, buffer;

    int epoch=0;





        //here's all OK;

        while(currentError>ecrit)
        {
            randCounter = randArr[randCounter%863]%787;
            currentError = 0.0;
            //mix vectors
            for(int i=0; i<5*NumberOfVectors; ++i)
            {
                a1 = randArr[(randCounter++)%953]%(NumberOfVectors);
                a2 = randArr[(randCounter++)%911]%(NumberOfVectors);
                buffer=mixNum[a2];
                mixNum[a2]=mixNum[a1];
                mixNum[a1]=buffer;
            }
            //printf mixNum
            for(int i = 0; i < NumberOfVectors; ++i)
            {
//                printf("%d ", mixNum[i]);
            }
//            printf("\n\n");

            for(int vecNum = 0; vecNum < NumberOfVectors; ++vecNum)
            {
                if( mixNum[vecNum] == get_global_id(0) ) continue; //not to learn with h'th vector

                type = round(matrix[mixNum[vecNum] * (NetLength+2) + NetLength+1]);

                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    output[j]=0.;
                    for(int i=0; i<NetLength+1; ++i)   // +bias, coz +1
                    {
                        output[j] += weight[j * (NetLength+1) + i] * matrix[ mixNum[vecNum] * (NetLength+2) + i ]; //mixNum problem - printf
                    }

                    output[j] = 1. / ( 1. + exp(-output[j] / temp) );
                }

                //error count + weight differ
                currentError+=(1.-output[type])*(1.-output[type]);
                for(int i=0; i<NumOfClasses; ++i)
                {
                    if(i!=type)
                    {
                        currentError += output[i] * output[i];
                    }
                }
                //learn itself
                for(int i = 0; i < (NetLength+1); ++i)
                {
                    weight[type * (NetLength+1) + i] += lrate * (1.-output[type]) * matrix[mixNum[vecNum] * (NetLength+2) + i];
                    for(int k=0; k<NumOfClasses; ++k)
                    {
                        if (k!=type)
                        {
                            weight[k * (NetLength+1) + i] -= lrate * output[k] * matrix[mixNum[vecNum] * (NetLength+2) + i];
                        }
                    }
                }
            }
            currentError/=NumberOfVectors;
            currentError=sqrt(currentError);
            ++epoch;
        }

}/*
    type = matrix[get_global_id(0) * (NetLength+2) + NetLength+1];
    for(int j = 0; j < NumOfClasses; ++j) //calculate output //2 = numberOfTypes
    {
        output[j] = 0.;
        for(int i = 0; i < NetLength; ++i)
        {
            output[j]+=weight[j * (NetLength+1) + i] * matrix[get_global_id(0) * (NetLength+2) + i];
        }
        output[j] += weight[j * (NetLength+1) + NetLength] * matrix[get_global_id(0) * (NetLength+2) + NetLength];
    //    output[j] = logistic(output[j], temp); // unlinear conformation
        output[j] = 1. / ( 1. + exp(-output[j] / temp) );
    }
    bool right = 1;
    float outp = output[type];
    for(int k = 0; k < NumOfClasses; ++k)
    {
        if(k != type && output[k] >= outp)
        {
            right = false;
            outp = output[k];
        }
    }
    if(!right && matrix[get_global_id(0) * (NetLength+2) + NetLength+1]!=1.5) ++NumberOfErrors[type]; //generality
    outError[get_global_id(0)] = 0.;
    for(int k = 0; k < NumOfClasses; ++k)
    {
        if(k!=type)
        {
            outError[get_global_id(0)] += (output[k] * output[k]);
        }
        else
        {
            outError[get_global_id(0)] += (1. - output[k]) * (1. - output[k]);
        }
    }
    outError[get_global_id(0)] = sqrt(outError[get_global_id(0)]);
    answer[get_global_id(0)] = right; //return value

}

*/
