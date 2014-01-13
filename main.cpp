// A minimalist OpenCL program.
#include <CL/cl.h>
#include <stdio.h>
#include <iostream>
#define NWITEMS 512

#pragma OPENCL EXTENSION cl_khr_icd : enable

using namespace std;


// A simple memset kernel
const char *source =
        "__kernel void memset( __global uint *dst ) \n"
        "{ \n"
        " dst[get_global_id(0)] = get_global_id(0); \n"
        "} \n";

int errorMessage(cl_int error_)
{
    return (int)error_;
}

int main(int argc, char ** argv)
{
    cl_int error = 0;


    // 1. Get a platform.
    cl_platform_id platform;
    cl_uint numOfPlatforms;
    CL_INVALID_VALUE;
    CL_OUT_OF_HOST_MEMORY;
//    CL_PLATFORM_NOT_FOUND_KHR;
    CL_FALSE;
    error = clGetPlatformIDs( 1, &platform, &numOfPlatforms );
    cout << "numOfPlatforms = " << numOfPlatforms << endl;
    if(error != CL_SUCCESS)
    {
       cout << "Error getting platform id: " << errorMessage(error) << endl;
       exit(error);
    }


    // 2. Find a gpu device.
    cl_device_id device;
    error = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU,
                    1,
                    &device,
                    NULL); //error
    if(error != CL_SUCCESS)
    {
       cout << "Error getting device ids: " << errorMessage(error) << endl;
       exit(error);
    }


    // 3. Create a context and command queue on that device.
    cl_context context = clCreateContext( NULL,
                                          1,
                                          &device,
                                          NULL, NULL, &error);
    if(error != CL_SUCCESS)
    {
       cout << "Error creating context: " << errorMessage(error) << endl;
       exit(error);
    }


    cl_command_queue queue = clCreateCommandQueue( context,
                                                   device,
                                                   0, &error );
    if(error != CL_SUCCESS)
    {
       cout << "Error creating command queue: " << errorMessage(error) << endl;
       exit(error);
    }


    // 4. Perform runtime source compilation, and obtain kernel entry point.
    cl_program program = clCreateProgramWithSource( context,
                                                    1,
                                                    &source,
                                                    NULL, NULL );
    clBuildProgram( program, 1, &device, NULL, NULL, NULL );
    cl_kernel kernel = clCreateKernel( program, "memset", NULL );


    // 5. Create a data buffer.
    cl_mem buffer = clCreateBuffer( context,
                                    CL_MEM_WRITE_ONLY,
                                    NWITEMS * sizeof(cl_uint),
                                    NULL, NULL );
    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = NWITEMS;
    clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);
    clEnqueueNDRangeKernel( queue,
                            kernel,
                            1,
                            NULL,
                            &global_work_size,
                            NULL, 0, NULL, NULL);
    clFinish( queue );
    // 7. Look at the results via synchronous buffer map.
    cl_uint *ptr;
    ptr = (cl_uint *) clEnqueueMapBuffer( queue,
                                          buffer,
                                          CL_TRUE,
                                          CL_MAP_READ,
                                          0,
                                          NWITEMS * sizeof(cl_uint),
                                          0, NULL, NULL, NULL );
    cout << "wow" << endl;

    for(int i = 0; i < NWITEMS; i++)
    {
        cout << i << '\t' << ptr[i] << endl;
        printf("%d %d\n", i, ptr[i]);
    }
    return 0;
}
