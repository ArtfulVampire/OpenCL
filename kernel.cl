#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void leaveOneOut(
constant double * params0,
global double * matrix, //NumberOfVectors * (NetLength+2),
constant int * params1,
global double * weight, //NumberOfClasses * (NetLength+1)
/*private*/ global int * mixNum,
/*private*/ global double * output,
global int * answer,
global double * outError,
/*private*/ global double * outputClass,
global double * NumberOfErrors,
constant int * randArr)
{

    double ecrit = params0[0];
    double lrate = params0[1];
    double temp = params0[2];
    int NumberOfVectors = params1[0];
    int NumOfClasses = params1[1];
    int NetLength = params1[2];

    double currentError = 2. * ecrit;
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
            //        cout<<"epoch="<<epoch<<endl;

            for(int vecNum = 0; vecNum < NumberOfVectors; ++vecNum)
            {
                if( mixNum[vecNum] == get_global_id(0) ) continue; //not to learn with h'th vector

                type = round(matrix[mixNum[vecNum] * (NetLength+2) + NetLength+1]);

                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    output[j]=0.;
                    for(int i=0; i<NetLength+1; ++i)   // +bias, coz +1
                    {
                        output[j]+=weight[j * (NetLength+1) + i]*matrix[mixNum[vecNum] * (NetLength+2) + i];
                    }
//                    output[j] = logistic(output[j], temp); // unlinear logistic conformation
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
}
/*

    type = matrix[get_global_id(0) * (NetLength+2) + NetLength+1];
    for(int j = 0; j < NumOfClasses; ++j) //calculate output //2 = numberOfTypes
    {
        outputClass[j] = 0.;
        for(int i = 0; i < NetLength; ++i)
        {
            outputClass[j]+=weight[j * (NetLength+1) + i] * matrix[get_global_id(0) * (NetLength+2) + i];
        }
        outputClass[j] += weight[j * (NetLength+1) + NetLength] * matrix[get_global_id(0) * (NetLength+2) + NetLength];
    //    outputClass[j] = logistic(outputClass[j], temp); // unlinear conformation
        outputClass[j] = 1. / ( 1. + exp(-outputClass[j] / temp) );
    }
    bool right = 1;
    double outp = outputClass[type];
    for(int k = 0; k < NumOfClasses; ++k)
    {
        if(k != type && outputClass[k] >= outp)
        {
            right = false;
            outp = outputClass[k];
        }
    }
    if(!right && matrix[get_global_id(0) * (NetLength+2) + NetLength+1]!=1.5) ++NumberOfErrors[type]; //generality
    outError[get_global_id(0)] = 0.;
    for(int k = 0; k < NumOfClasses; ++k)
    {
        if(k!=type)
        {
            outError[get_global_id(0)] += (outputClass[k] * outputClass[k]);
        }
        else
        {
            outError[get_global_id(0)] += (1. - outputClass[k]) * (1. - outputClass[k]);
        }
    }
    outError[get_global_id(0)] = sqrt(outError[get_global_id(0)]);
    answer[get_global_id(0)] = right; //return value
}
*/
