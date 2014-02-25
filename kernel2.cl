#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float logistic(float x, float t)
{
    if( x >   37.*t )  return 1.;
    if( x < -115.*t )  return 0.;
    return 1. / ( 1. + exp(-x/t) );
}
//each thread deals with his input dimension

//get_work_dim()
//get_global_size(0)
//get_local_size(0)
//get_num_groups(0)
//get_global_id(0)
//get_local_id(0)

__kernel void leaveOneOut(
constant float * params0,
constant float * matrix, //NumberOfVectors * (NetLength+2),
constant int * params1,
global float * weight, //NumberOfClasses * (NetLength+1)
global float *currentError,
global float *sync,

global int * mixNum,
global float * output,

constant int * randArr)
{


    float ecrit = params0[0]; //ok
    float lrate = params0[1]; //ok
    float temp = params0[2]; //OK



    int NumberOfVectors = params1[0]; //ok
    int NumOfClasses = params1[1]; //ok
    int NetLength = params1[2]; //ok
    int NumToSkip = params1[3];



    local float data[128]; //private
    for(int j = 0; j < NumberOfVectors; ++j)
    {
        data[j] = matrix[j * (NetLength + 2) + get_global_id(0)];
    }

    private int type;

    private int randCounter;
    randCounter = 12*get_global_id(0);



//    printf("%d\n", get_global_id(0));
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    return;


    if(get_global_id(0) == 0)
    {
        *currentError = 2. * ecrit;
        printf("NumberOfVectors = %d\n", NumberOfVectors);
        printf("currentError = %f\n", *currentError);
        printf("global_size = %d\n", get_global_size(0));
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

    barrier(CLK_GLOBAL_MEM_FENCE);
//    return;
//    printf("%.3f\t", data[0]);
//    barrier(CLK_GLOBAL_MEM_FENCE);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    return;



    int a1, a2, buffer;

    local short int epoch;
    epoch = 0;
        while(*currentError > ecrit)
        {

            if(get_global_id(0) == 0)
            {

                randCounter = randArr[randCounter%863]%787;
                *currentError = 0.0f;
                for(int i=0; i<5*NumberOfVectors; ++i)
                {
                    a1 = randArr[(randCounter++)%953]%(NumberOfVectors);
                    a2 = randArr[(randCounter++)%911]%(NumberOfVectors);
                    buffer=mixNum[a2];
                    mixNum[a2]=mixNum[a1];
                    mixNum[a1]=buffer;
                }

                for(int i=0; i<NumberOfVectors; ++i)
                {
                    printf("%d ", mixNum[i]);
                }
                printf("\n");
                sync = 1;
            }
            else
            {

            }
            barrier(CLK_GLOBAL_MEM_FENCE);


            for(int vecNum = 0; vecNum < NumberOfVectors; ++vecNum)
            {
                if( mixNum[vecNum] == NumToSkip )
                {
                    continue; //not to learn with h'th vector
                }
                printf("numThr = %d\tmixNum = %d\n", get_global_id(0), mixNum[vecNum]);
                return;

                type = round(matrix[mixNum[vecNum] * (NetLength+2) + NetLength+1]); //OK
                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    if(get_global_id(0) == 0) output[j] = 0.;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);

                for(int j = 0; j < NumOfClasses; ++j) //calculate output
                {
                printf("%.3f\n", data[mixNum[vecNum]]);
//                    output[j] += weight[j * (NetLength+1) + get_global_id(0)] * data[mixNum[vecNum]]; //error
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
                return;

                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    if(get_global_id(0) == 0) output[j] = logistic(output[j], temp);
                }
                barrier(CLK_GLOBAL_MEM_FENCE);

                weight[type * (NetLength+1) + get_global_id(0)] += lrate * (1.-output[type]) * data[ mixNum[vecNum] ];
                for(int k=1; k<NumOfClasses; ++k)
                {
                    weight[((type+k)%NumOfClasses) * (NetLength+1) + get_global_id(0)] -= lrate * output[((type+k)%NumOfClasses)] * data[mixNum[vecNum]];
                }

                barrier(CLK_GLOBAL_MEM_FENCE);
            }

            if(get_global_id(0) == 0)
            {

                *currentError += (1.-output[type])*(1.-output[type]);
                for(int i=1; i<NumOfClasses; ++i)
                {
                    *currentError += output[(type+i)%NumOfClasses] * output[(type+i)%NumOfClasses];
                }
                *currentError/=NumberOfVectors;
                *currentError=sqrt(*currentError);
                printf("epoch = %d\terror = %f\n", epoch, *currentError);
                ++epoch;
            }
//            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);

        }
}


