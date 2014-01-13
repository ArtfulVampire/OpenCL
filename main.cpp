// A minimalist OpenCL program.
#include <CL/cl.h>
#include <stdio.h>
#include <iostream>
#include <QTime>
#include <ctime>
#define NWITEMS 512

#pragma OPENCL EXTENSION cl_khr_icd : enable

using namespace std;

int errorMessage(cl_int error_);
const char* kernelFromFile(char* path);

int main(int argc, char ** argv)
{
    cl_int error = 0;
    cl_platform_id platform;
    cl_uint numOfPlatforms;
    int dev, nw;
    int NDEVS = 2;
    cl_device_type devs[NDEVS];
    devs[0] = CL_DEVICE_TYPE_CPU;
    devs[1] = CL_DEVICE_TYPE_GPU;
    cl_uint *src_ptr;
    unsigned int num_src_items = 4096*4096;


//    time(srand(QTime::currentTime().msec()));
    time_t ltime;
    time(&ltime);

    src_ptr = new cl_uint [num_src_items];
    cl_uint a = (cl_uint)ltime, b = (cl_uint)ltime;
    cl_uint min = (cl_uint) -1;

    for( int i=0; i < num_src_items; i++ )
    {
        src_ptr[i] = (cl_uint) (b = ( a * ( b & 65535 )) + ( b >> 16 ));
        min = src_ptr[i] < min ? src_ptr[i] : min;
    }

    error = clGetPlatformIDs( 1, &platform, &numOfPlatforms );
    if(error != CL_SUCCESS)
    {
       cout << "Error getting platform id: " << errorMessage(error) << endl;
       exit(error);
    }


//    // 2. Find a gpu device.
//    cl_device_id device;

//    error = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU,
//                    1,
//                    &device,
//                    NULL); //error
//    if(error != CL_SUCCESS)
//    {
//       cout << "Error getting device ids: " << errorMessage(error) << endl;
//       exit(error);
//    }


//    // 3. Create a context and command queue on that device.
//    cl_context context = clCreateContext( NULL,
//                                          1,
//                                          &device,
//                                          NULL, NULL, &error);
//    if(error != CL_SUCCESS)
//    {
//       cout << "Error creating context: " << errorMessage(error) << endl;
//       exit(error);
//    }


//    cl_command_queue queue = clCreateCommandQueue( context,
//                                                   device,
//                                                   0, &error );
//    if(error != CL_SUCCESS)
//    {
//       cout << "Error creating command queue: " << errorMessage(error) << endl;
//       exit(error);
//    }

//    const char *source = (const char*)kernelFromFile("/home/michael/Qt/Projects/myOpenCL/kernel.cl");

//    // 4. Perform runtime source compilation, and obtain kernel entry point.
//    cl_program program = clCreateProgramWithSource( context,
//                                                    1,
//                                                    &source,
//                                                    NULL, NULL );
//    clBuildProgram( program, 1, &device, NULL, NULL, NULL );

//    cl_kernel kernel = clCreateKernel( program, "memset", NULL );


//    // 5. Create a data buffer.
//    cl_mem buffer = clCreateBuffer( context,
//                                    CL_MEM_WRITE_ONLY,
//                                    NWITEMS * sizeof(cl_uint),
//                                    NULL, NULL );

//    // 6. Launch the kernel. Let OpenCL pick the local work size.
//    size_t global_work_size = NWITEMS;
//    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);

//    QTime myTime;
//    myTime.start();

//    clEnqueueNDRangeKernel( queue,
//                            kernel,
//                            1,
//                            NULL,
//                            &global_work_size,
//                            NULL, 0, NULL, NULL);
//    clFinish( queue );

//    // 7. Look at the results via synchronous buffer map.
//    cl_uint *ptr;
//    ptr = (cl_uint *) clEnqueueMapBuffer( queue,
//                                          buffer,
//                                          CL_TRUE,
//                                          CL_MAP_READ,
//                                          0,
//                                          NWITEMS * sizeof(cl_uint),
//                                          0, NULL, NULL, NULL );

//    for(int i = 0; i < NWITEMS; i++)
//    {
//        cout << i << '\t' << ptr[i] << endl;
//    }
//    cout << "time elapsed = " << myTime.elapsed() << endl;

    for(int dev = 0; dev < NDEVS; ++dev)
    {
        cout << "asdasd" << endl;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel minp;
        cl_kernel reduce;
        cl_mem src_buf;
        cl_mem dst_buf;
        cl_mem dbg_buf;
        cl_uint *dst_ptr,
                *dbg_ptr;
        cout << (dev == 0 ? "CPU" : "GPU") << endl;

        // Find the device.
        error = clGetDeviceIDs( platform,
                        devs[dev],
                        1,
                        &device,
                        NULL);
        if(error != CL_SUCCESS)
        {
            cout << "Error getting device ids: " << errorMessage(error) << endl;
            exit(error);
        }


        // 4. Compute work sizes.
        cl_uint compute_units;
        size_t global_work_size;
        size_t local_work_size;
        size_t num_groups;
        error = clGetDeviceInfo( device,
                         CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(cl_uint),
                         &compute_units,
                         NULL);

        if(error != CL_SUCCESS)
        {
            cout << "Error getting device info: " << errorMessage(error) << endl;
            exit(error);
        }

        if( devs[dev] == CL_DEVICE_TYPE_CPU )
        {
            global_work_size = compute_units * 1; // 1 thread per core
            local_work_size = 1;
        }
        else
        {
            cl_uint ws = 64;
            global_work_size = compute_units * 7 * ws; // 7 wavefronts per SIMD
            while( (num_src_items / 4) % global_work_size != 0 )
                global_work_size += ws;  //??????????????????????????????????
            local_work_size = ws;
        }
        num_groups = global_work_size / local_work_size;

        // Create a context and command queue on that device.
        context = clCreateContext( NULL,
                                   1,
                                   &device,
                                   NULL, NULL, NULL);
        queue = clCreateCommandQueue(context,
                                     device,
                                     0, NULL);
        // Minimal error check.
        if( queue == NULL )
        {
            cout << "Compute device setup failed" << endl; ;
            return(-1);
        }
        // Perform runtime source compilation, and obtain kernel entry point.
        const char *kernel_source = (const char*)kernelFromFile("/home/michael/Qt/Projects/myOpenCL/kernel.cl");
        cl_int ret = 0;
        program = clCreateProgramWithSource( context,
                                             1,
                                             &kernel_source,
                                             NULL, &ret );
        //Tell compiler to dump intermediate .il and .isa GPU files.
        // 5. Print compiler error messages
        if(ret != CL_SUCCESS)
        {
            cout << "clBuildProgram failed: " << errorMessage(ret) << endl;
            char buf[0x10000];
            clGetProgramBuildInfo( program,
                                   device,
                                   CL_PROGRAM_BUILD_LOG,
                                   0x10000,
                                   buf,
                                   NULL);
            printf("\n%s\n", buf);
            return(-1);
        }

        minp = clCreateKernel( program, "minp", &error );

        if (error != CL_SUCCESS)
        {
            cout << "Cannot create kernel minp: " << errorMessage(error) << endl;
            exit(-1);
        }
        reduce = clCreateKernel( program, "reduce", &error );
        if (error != CL_SUCCESS)
        {
            cout << "Cannot create kernel reduce: " << errorMessage(error) << endl;
            exit(-1);
        }

        // Create input, output and debug buffers.

        src_buf = clCreateBuffer( context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  num_src_items * sizeof(cl_uint),
                                  src_ptr,
                                  NULL );
        dst_buf = clCreateBuffer( context,
                                  CL_MEM_READ_WRITE,
                                  num_groups * sizeof(cl_uint),
                                  NULL, NULL );
        dbg_buf = clCreateBuffer( context,
                                  CL_MEM_WRITE_ONLY,
                                  global_work_size * sizeof(cl_uint),
                                  NULL, NULL );

        clSetKernelArg(minp, 0, sizeof(void *), (void*) &src_buf);
        clSetKernelArg(minp, 1, sizeof(void *), (void*) &dst_buf);
        clSetKernelArg(minp, 2, 1*sizeof(cl_uint), (void*) NULL);
        clSetKernelArg(minp, 3, sizeof(void *), (void*) &dbg_buf);
        clSetKernelArg(minp, 4, sizeof(num_src_items), (void*) &num_src_items);
        clSetKernelArg(minp, 5, sizeof(dev), (void*) &dev);
        clSetKernelArg(reduce, 0, sizeof(void *), (void*) &src_buf);
        clSetKernelArg(reduce, 1, sizeof(void *), (void*) &dst_buf);
        QTime myTime;
        myTime.start();
        // 6. Main timing loop.
#define NLOOPS 500

        cl_event ev;
        int nloops = NLOOPS;
        while(nloops--)
        {
            clEnqueueNDRangeKernel( queue,
                                    minp,
                                    1,
                                    NULL,
                                    &global_work_size,
                                    &local_work_size,
                                    0, NULL, &ev);
            clEnqueueNDRangeKernel( queue,
                                    reduce,
                                    1,
                                    NULL,
                                    &num_groups,
                                    NULL, 1, &ev, NULL);
        }
        clFinish( queue );

        printf("B/W %.2f GB/sec, ", ((float) num_src_items *
                                     sizeof(cl_uint) * NLOOPS) /
               double(myTime.elapsed()) / 1e9 );
        // 7. Look at the results via synchronous buffer map.
        dst_ptr = (cl_uint *) clEnqueueMapBuffer( queue,
                                                  dst_buf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  num_groups * sizeof(cl_uint),
                                                  0, NULL, NULL, NULL );
        dbg_ptr = (cl_uint *) clEnqueueMapBuffer( queue,
                                                  dbg_buf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  global_work_size *
                                                  sizeof(cl_uint),
                                                  0, NULL, NULL, NULL );
        // 8. Print some debug info.
        printf("%d groups, %d threads, count %d, stride %d\n", dbg_ptr[0],
                dbg_ptr[1],
                dbg_ptr[2],dbg_ptr[3] );
        if( dst_ptr[0] == min )
            printf("result correct\n");
        else
            printf("result INcorrect\n");
    }


    return 0;
}

const char* kernelFromFile(char* path)
{
    char* tempString = new char [200];
    char* shaderString = new char [3000];
    int currentIndex = 0;
    FILE * shad = fopen(path, "r");
    if(shad == NULL)
    {
        cout<<"Cannot open file\n"<<endl;
        return (const char*)NULL;
    }
    while(1)
    {

        fgets(tempString, 50, shad);
        if(feof(shad)) break;
        for(int i = 0; i < strlen(tempString); ++i)
        {
            shaderString[currentIndex++] = tempString[i];
        }
    }
    shaderString[currentIndex] = '\0';
    fclose(shad);

    delete []tempString;
    return shaderString;
}

int errorMessage(cl_int error_)
{
    return (int)error_;
}
