
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void leaveOneOut(
int NumberOfVectors,
local int * mixNum,
constant int * randArr)
{

printf("%d\n", get_global_id(0));

    int randCounter = 12*get_global_id(0);

    for(int i=0; i<NumberOfVectors; ++i)
    {
        mixNum[i]=i;
    }

    int a1, a2, buffer;

//            if(get_global_id(0) == 0)
//            {
//                for(int i = 0; i < NumberOfVectors; ++i)
//                {
//                    printf("%d ", mixNum[i]);
//                }
//                printf("\n\n");
//            }

            randCounter = randArr[randCounter%863]%787;
            //mix vectors
            for(int i=0; i<5*NumberOfVectors; ++i)
            {
                a1 = randArr[(randCounter++)%953]%(NumberOfVectors);
                a2 = randArr[(randCounter++)%911]%(NumberOfVectors);
                buffer=mixNum[a2];
                mixNum[a2]=mixNum[a1];
                mixNum[a1]=buffer;
            }
            for(int i = 0; i < NumberOfVectors; ++i)
            {
                printf("%d ", mixNum[i]);
            }
            printf("\n\n");

}
