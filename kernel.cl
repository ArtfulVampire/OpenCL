#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float logistic(float x, float t)
{
    if( x >   37.*t )  return 1.;
    if( x < -115.*t )  return 0.;
    return 1. / ( 1. + exp(-x/t) );
}

//each thread deals with his input vector (~150)

__kernel void leaveOneOut(
constant float * params0,
global float * matrix, //NumberOfVectors * (NetLength+2),
constant int * params1,
global float * weight, //NumberOfClasses * (NetLength+1)

local int * mixNum,

constant int * randArr)
{

    float ecrit = params0[0]; //ok
    float lrate = params0[1]; //ok
    float temp = params0[2]; //OK

    int NumberOfVectors = params1[0]; //ok
    int NumOfClasses = params1[1]; //ok
    int NetLength = params1[2]; //ok
    int NumToSkip = params1[3];

    private float output[3];


    local float currentError;
    currentError = 2. * ecrit;
    private int type;

    private int randCounter;
    randCounter = 12*get_global_id(0);



    if(get_global_id(0) == 0)
    {
        for(int j = 0; j<NumOfClasses; ++j) //calculate output
        {
            for(int i = 0; i < NetLength+1; ++i)
            {
                weight[j * (NetLength+1) + i] = 0.;
            }
        }

        for(int i=0; i<NumberOfVectors; ++i)
        {
            mixNum[i]=i;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);




    int a1, a2, buffer;

    local int epoch;
    epoch = 0;
        while(currentError > ecrit)
        {

            barrier(CLK_LOCAL_MEM_FENCE);
            if(get_global_id(0) == 0)
            {

                randCounter = randArr[randCounter%863]%787;
                currentError = 0.0;
                for(int i=0; i<5*NumberOfVectors; ++i)
                {
                    a1 = randArr[(randCounter++)%953]%(NumberOfVectors);
                    a2 = randArr[(randCounter++)%911]%(NumberOfVectors);
                    buffer=mixNum[a2];
                    mixNum[a2]=mixNum[a1];
                    mixNum[a1]=buffer;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for(int vecNum = get_global_id(0); vecNum < NumberOfVectors; vecNum += get_global_size(0) )
            {
                if( mixNum[vecNum] == NumToSkip ) continue; //not to learn with h'th vector



                type = round(matrix[mixNum[vecNum] * (NetLength+2) + NetLength+1]); //OK

                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    //error somewhere here
                    output[j]=0.;
                    for(int i = 0; i < NetLength+1; ++i)
                    {
                        output[j] += weight[j * (NetLength+1) + i] * matrix[ mixNum[vecNum] * (NetLength+2) + i ];
                    }
                    output[j] = logistic(output[j], temp);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                //error count + weight differ
                currentError += (1.-output[type])*(1.-output[type]);
                for(int i=0; i<NumOfClasses; ++i)
                {
                    if(i!=type)
                    {
                        currentError += output[i] * output[i];
                    }
                }
                //learn itself

                barrier(CLK_LOCAL_MEM_FENCE);


                for(int i = 0; i < (NetLength+1); ++i)
                {
                    weight[type * (NetLength+1) + i] += lrate * (1.-output[type]) * matrix[ mixNum[vecNum] * (NetLength+2) + i];
                    for(int k=0; k<NumOfClasses; ++k)
                    {
                        if (k!=type)
                        {
                            weight[k * (NetLength+1) + i] -= lrate * output[k] * matrix[mixNum[vecNum] * (NetLength+2) + i];
                        }
                    }
                }

            }

            barrier(CLK_LOCAL_MEM_FENCE);
            if(get_global_id(0) == 0)
            {
                currentError/=NumberOfVectors;
                currentError=sqrt(currentError);
                printf("epoch = %d\terror = %f\n", epoch, currentError);
                ++epoch;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

        }
}


