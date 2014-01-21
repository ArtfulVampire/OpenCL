#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void memset( __global uint *dst )
{
dst[get_global_id(0)] = get_global_id(0);
}

__kernel void minp( __global uint4 *src,
__global uint *gmin,
__local uint *lmin,
__global uint *dbg,
int nitems,
uint dev )
{
// 10. Set up __global memory access pattern.

    uint count = ( nitems / 4 ) / get_global_size(0);
    uint idx = (dev == 0) ? get_global_id(0) * count : get_global_id(0);
    uint stride = (dev == 0) ? 1 : get_global_size(0);
    uint pmin = (uint) -1;

 // 11. First, compute private min, for this work-item.

    for( int n=0; n < count; n++, idx += stride )
    {
        pmin = min( pmin, src[idx].x );
        pmin = min( pmin, src[idx].y );
        pmin = min( pmin, src[idx].z );
        pmin = min( pmin, src[idx].w );
    }

 // 12. Reduce min values inside work-group.

    if( get_local_id(0) == 0 )
    lmin[0] = (uint) -1;

    barrier( CLK_LOCAL_MEM_FENCE );

    (void) atom_min( lmin, pmin );

    barrier( CLK_LOCAL_MEM_FENCE );

 // Write out to __global.

    if( get_local_id(0) == 0 )
    gmin[ get_group_id(0) ] = lmin[0];

 // Dump some debug information.

    if( get_global_id(0) == 0 )
    {
        dbg[0] = get_num_groups(0);
        dbg[1] = get_global_size(0);
        dbg[2] = count;
        dbg[3] = stride;
    }
 }

// 13. Reduce work-group min values from __global to __global.

__kernel void reduce( __global uint4 *src,
__global uint *gmin )
{
    (void) atom_min( gmin, gmin[get_global_id(0)] ) ;
}


double logistic(double x, double t)
{
    if( x >   37.*t )  return 1.;
    if( x < -115.*t )  return 0.;
    return 1. / ( 1. + exp(-x/t) );
}


__kernel void leaveOneOut(
constant double * params0,
global double * matrix, //NumberOfVectors * (NetLength+2),
constant int * params1,
private double * weight, //NumberOfClasses * (NetLength+1)
private int * mixNum,
private double * output,
global int * answer,
global double * outError,
private double * outputClass,
global double * NumberOfErrors,
global int * NumOfThread,
global int * NumOfVectorToSkip,
constant int * randArr)
{

    double ecrit = params0[0];
    double lrate = params0[1];
    double temp = params0[2];
    int NumberOfVectors = params1[0];
    int NumOfClasses = params1[1];
    int NetLength = params1[2];

    int vecNum;
    double currentError = 2. * ecrit;
    int type=0;
    int randCounter = 12*get_global_id(0);

    NumOfThread[get_global_id(0)] = get_global_id(0);

    NumOfVectorToSkip[get_global_id(0)] = get_global_id(0);/////////
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

            for(vecNum = 0; vecNum < NumberOfVectors; ++vecNum)
            {
                if( mixNum[vecNum] == NumOfVectorToSkip[get_global_id(0)] ) continue; //not to learn with h'th vector

                continue;
                type = round(matrix[mixNum[vecNum] * (NetLength+2) + NetLength+1]);

                for(int j = 0; j<NumOfClasses; ++j) //calculate output
                {
                    output[j]=0.;
                    for(int i=0; i<NetLength+1; ++i)   // +bias, coz +1
                    {
                        output[j]+=weight[j * (NetLength+1) + i]*matrix[mixNum[vecNum] * (NetLength+2) + i];
                    }
                    output[j] = logistic(output[j], temp); // unlinear logistic conformation
//                    output[j] = 1. / ( 1. + exp(-output[j] / temp) );
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
            break;
            currentError/=NumberOfVectors;
            currentError=sqrt(currentError);
            ++epoch;
        }
}
/*

    type = matrix[NumOfVectorToSkip[get_global_id(0)] * (NetLength+2) + NetLength+1];
    for(int j = 0; j < NumOfClasses; ++j) //calculate output //2 = numberOfTypes
    {
        outputClass[j] = 0.;
        for(int i = 0; i < NetLength; ++i)
        {
            outputClass[j]+=weight[j * (NetLength+1) + i] * matrix[NumOfVectorToSkip[get_global_id(0)] * (NetLength+2) + i];
        }
        outputClass[j] += weight[j * (NetLength+1) + NetLength] * matrix[NumOfVectorToSkip[get_global_id(0)] * (NetLength+2) + NetLength];
        outputClass[j] = logistic(outputClass[j], temp); // unlinear conformation
    //    outputClass[j] = 1. / ( 1. + exp(-outputClass[j] / temp) );
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
    if(!right && matrix[NumOfVectorToSkip[get_global_id(0)] * (NetLength+2) + NetLength+1]!=1.5) ++NumberOfErrors[type]; //generality
    outError[NumOfVectorToSkip[get_global_id(0)]] = 0.;
    for(int k = 0; k < NumOfClasses; ++k)
    {
        if(k!=type)
        {
            outError[NumOfVectorToSkip[get_global_id(0)]] += (outputClass[k] * outputClass[k]);
        }
        else
        {
            outError[NumOfVectorToSkip[get_global_id(0)]] += (1. - outputClass[k]) * (1. - outputClass[k]);
        }
    }
    outError[NumOfVectorToSkip[get_global_id(0)]] = sqrt(outError[NumOfVectorToSkip[get_global_id(0)]]);
    answer[NumOfVectorToSkip[get_global_id(0)]] = right; //return value
}
*/
