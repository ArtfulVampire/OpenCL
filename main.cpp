#include <CL/cl.h>
#include <stdio.h>
#include <iostream>
#include <QTime>
#include <ctime>
#include <cmath>
#define NWITEMS 512

using namespace std;

int errorMessage(cl_int error_);
const char* kernelFromFile(char* path);
/*
int main(int argc, char ** argv)
{
    cl_int error = 0;
    cl_platform_id platform;
    cl_uint numOfPlatforms;
    cl_int compute_units;
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


    error = clGetPlatformIDs( 1, &platform, NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Error getting platform id: " << error << endl;
       exit(error);
    }


    // 2. Find a gpu device.
    cl_device_id device;

    error = clGetDeviceIDs( platform, CL_DEVICE_TYPE_CPU,
                    1,
                    &device,
                    NULL); //error
    if(error != CL_SUCCESS)
    {
       cout << "Error getting device ids: " << errorMessage(error) << endl;
       exit(error);
    }

    error = clGetDeviceInfo( device,
                             CL_DEVICE_MAX_COMPUTE_UNITS,
                             sizeof(cl_uint),
                             &compute_units,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot count compute units: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "Number of Compute units = " << compute_units << endl;
    }

    cl_uint max_dim;
    error = clGetDeviceInfo( device,
                             CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                             sizeof(cl_uint),
                             &max_dim,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot count dimensions: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "Maximal dimensionality = " << max_dim << endl;
    }

    char* builtInKernels = new char [10000];
    error = clGetDeviceInfo( device,
                             CL_DEVICE_BUILT_IN_KERNELS,
                             sizeof(char)*10000,
                             builtInKernels,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot get builtInKernels: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "builtInKernels = " << builtInKernels << endl;
    }
    delete []builtInKernels;

    cl_device_fp_config doubleSupp;
    error = clGetDeviceInfo( device,
                             CL_DEVICE_DOUBLE_FP_CONFIG,
                             sizeof(cl_device_fp_config),
                             (void*) &doubleSupp,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot get double support: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "double support = " << doubleSupp << endl;
        CL_FP_DENORM;
        CL_FP_INF_NAN;
        CL_FP_ROUND_TO_NEAREST;
        CL_FP_ROUND_TO_ZERO;
        CL_FP_ROUND_TO_INF;
//        CP_FP_FMA;
        CL_FP_SOFT_FLOAT;
    }

    char* extensions = new char [10000];
    error = clGetDeviceInfo( device,
                             CL_DEVICE_EXTENSIONS,
                             sizeof(char) * 10000,
                             extensions,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot get extensions: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "extensions = " << extensions << endl;
    }
    delete []extensions;


    char* version = new char [300];
    error = clGetDeviceInfo( device,
                             CL_DEVICE_OPENCL_C_VERSION,
                             sizeof(char) * 300,
                             version,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot get version: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "version = " << version << endl;
    }
    delete []version;


    char* deviceName = new char [300];
    error = clGetDeviceInfo( device,
                             CL_DEVICE_NAME,
                             sizeof(char) * 300,
                             deviceName,
                             NULL);
    if(error != CL_SUCCESS)
    {
       cout << "Cannot get deviceName: " << errorMessage(error) << endl;
       exit(error);
    }
    else
    {
        cout << "deviceName = " << deviceName << endl;
    }
    delete []deviceName;


//    return 0;



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

    const char *source = (const char*)kernelFromFile("/home/michael/Qt/Projects/myOpenCL/kernel.cl");

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

//    cout << "wat" << endl;
    // 6. Launch the kernel. Let OpenCL pick the local work size.
    size_t global_work_size = NWITEMS;

    clSetKernelArg(kernel, 0, sizeof(void *), (void*) &buffer);

    QTime myTime;
    myTime.start();

    error = clEnqueueNDRangeKernel( queue,
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

    for(int i = 0; i < NWITEMS; i++)
    {
        cout << i << '\t' << ptr[i] << endl;
    }
    cout << "time elapsed = " << myTime.elapsed() << endl;



    return 0;
}
*/

void kernelFromFile(char * shaderString, char* path)
{
    char* tempString = new char [500];
    int currentIndex = 0;
    FILE * shad = fopen(path, "r");
    if(shad == NULL)
    {
        cout<<"Cannot open file\n"<<endl;
        return;
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
}

int errorMessage(cl_int error_)
{
    return (int)error_;
}







int main()
{
    double ** matrix;
    int NumberOfVectors = 0;
    int NetLength = 247*19;
    int ns = 19;
    int spLength = 247;
    int NumOfClasses = 3;
    QString helpString;
    double ecrit = 0.1;
    double lrate = 0.1;
    double temp = 10.;
    double Error = 0.;
    int * NumberOfErrors;
    int * randArr = new int [1000];
    srand(QTime::currentTime().msec());
    for(int i = 0; i < 1000; ++i)
    {
        randArr[i] = rand()%67718;
    }




    helpString = "/media/Files/Data/AAX/PA/all.pa";
    FILE *paSrc=fopen(helpString.toStdString().c_str(), "r");
    if(paSrc==NULL)
    {
        cout<<"pa-file==NULL"<<endl;
//        QMessageBox::critical((QWidget*)this, tr("Warning"), tr("cannot open pa-file"), QMessageBox::Ok);
        return -1;
    }
    NumberOfVectors=6000; //generality

    matrix = new double * [NumberOfVectors];
    for(int i=0; i<NumberOfVectors; ++i)
    {
        matrix[i] = new double[NetLength+2];
    }
    int num=0;
    double g[3];  //generality

    char ** FileName = new char* [NumberOfVectors];
    for(int i=0; i<NumberOfVectors; ++i)
    {
        FileName[i] = new char[40];
    }

//    cout<<"start pa-reading"<<endl;
    while(!feof(paSrc))
    {
        fscanf(paSrc, "%s\n", FileName[num]);  //read FileName

        for(int i=0; i<NetLength; ++i)
        {
            fscanf(paSrc, "%lf", &matrix[num][i]);
//            matrix[num][i]*=20;
        }

        if(NumOfClasses==3) fscanf(paSrc, "%lf %lf %lf\n", &g[0], &g[1], &g[2]); //read the class
        if(NumOfClasses==2)
        {
            fscanf(paSrc, "%lf %lf\n", &g[0], &g[1]);
            g[2]=0.;
//            cout<<"g[0]="<<g[0]<<" g[1]="<<g[1]<<" g[2]="<<g[2]<<endl;
        }

        matrix[num][NetLength]=1.; //bias
        matrix[num][NetLength+1]=0.*g[0] + 1.*g[1] + 2.*g[2]; //type
        if(matrix[num][NetLength+1]!=0. && matrix[num][NetLength+1]!=1. && matrix[num][NetLength+1]!=2. && matrix[num][NetLength+1]!=1.5)
        {
            cout<<"type is wrong "<<matrix[num][NetLength+1]<<endl;
            return -2;
        }
        ++num;
    }
    for(int i=num; i<NumberOfVectors; ++i)
    {
        delete []matrix[i];
        delete []FileName[i];
    }
    fclose(paSrc);
    NumberOfVectors=num;


    QTime myTime;
    myTime.start();
    cout << "leaveOneOutCL started" << endl;













    NumberOfErrors = new int[NumOfClasses];
    helpString="";
    for(int i=0; i<NumOfClasses; ++i)
    {
        NumberOfErrors[i]=0;
    }

    cl_int clError;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel leaveOneOutKernel;
    cl_device_type devType;
    cl_platform_id platform;


    devType = CL_DEVICE_TYPE_CPU;

    clError = clGetPlatformIDs(1,
                               &platform,
                               NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Cannot get platform Id: " << errorMessage(clError) << endl;
        exit(clError);
    }


    // Find the device.
    clError = clGetDeviceIDs( platform,
                    devType,
                    1,
                    &device,
                    NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Error getting device ids: " << errorMessage(clError) << endl;
        exit(clError);
    }

//    cl_device_fp_config doubleSupport;
//    cl_int doubleWork = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &doubleSupport, NULL);
//    cout << doubleWork << "\tDoubleSupport = " << doubleSupport << endl;
//    return -2;

    CL_INVALID_DEVICE;// if device is not valid.
    CL_INVALID_VALUE;// if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type as shown in the table above and param_value is not a NULL value or if param_name is a value that is available as an extension and the corresponding extension is not supported by the device.
    CL_OUT_OF_RESOURCES;// if there is a failure to allocate resources required by the OpenCL implementation on the device.
    CL_OUT_OF_HOST_MEMORY;// if there is a failure to allocate resources required by the OpenCL implementation on the host.

    // 4. Compute work sizes.
    cl_uint compute_units;
    size_t global_work_size;
    clError = clGetDeviceInfo(device,
                     CL_DEVICE_MAX_COMPUTE_UNITS,
                     sizeof(cl_uint),
                     &compute_units,
                     NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Error getting device info: " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "Max compute units = " << compute_units << endl;

    cl_ulong maxConstBufSize;
    clError = clGetDeviceInfo(device,
                     CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                     sizeof(cl_ulong),
                     &maxConstBufSize,
                     NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Error getting device info: " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "Max const buffer size = " << maxConstBufSize << " bytes"<< endl;
    cout << "it is = " << maxConstBufSize/sizeof(double) << " doubles"<< endl;

    cl_ulong globalMemSize;
    clError = clGetDeviceInfo(device,
                     CL_DEVICE_GLOBAL_MEM_SIZE,
                     sizeof(cl_ulong),
                     &globalMemSize,
                     NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Error getting device info: " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "Max globalMemSize = " << globalMemSize << " bytes"<< endl;
    cout << "it is = " << globalMemSize/1024./1024./1024. << " GB"<< endl;

    cl_ulong localMemSize;
    clError = clGetDeviceInfo(device,
                     CL_DEVICE_LOCAL_MEM_SIZE,
                     sizeof(cl_ulong),
                     &localMemSize,
                     NULL);
    if(clError != CL_SUCCESS)
    {
        cout << "Error getting device info: " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "Max localMemSize = " << localMemSize << " bytes"<< endl;
    cout << "it is = " << localMemSize/sizeof(double) << " doubles"<< endl;


//    if(compute_units > NumberOfVectors)
//    {
//        global_work_size = NumberOfVectors;
//        local_work_size = 1;
//    }
//    else
//    {
//        local_work_size = ceil(double(NumberOfVectors)/compute_units);
//        global_work_size = NumberOfVectors;
//    }


    global_work_size = NumberOfVectors;

    global_work_size = 8;



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
        return -1;
    }

    // Perform runtime source compilation, and obtain kernel entry point.
    char * kernel_source0 = new char [20000];
    kernelFromFile(kernel_source0, "/home/michael/Qt/Projects/myOpenCL/kernel.cl");
//    kernelFromFile(kernel_source0, "/home/michael/Qt/Projects/myOpenCL/kernel2.cl");
    const char *kernel_source = (const char*)kernel_source0;
    program = clCreateProgramWithSource( context,
                                         1,
                                         &kernel_source,
                                         NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create program : " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "start build program" << endl;

    clError = clBuildProgram( program, 1, &device, NULL, NULL, NULL); //read the options
    char *buf = new char [0x10000];
    clGetProgramBuildInfo( program,
                           device,
                           CL_PROGRAM_BUILD_LOG,
                           0x10000,
                           buf,
                           NULL);
    cout << buf << endl;
    delete []buf;

    if(clError != CL_SUCCESS)
    {
        exit(clError);
    }

    leaveOneOutKernel = clCreateKernel( program, "leaveOneOut", &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create kernel leaveOneOut: " << errorMessage(clError) << endl;
        exit(clError);
    }
    cout << "all the preparations done, memory allocation start, elapsed " << myTime.elapsed()/1000. << " sec" << endl;
    myTime.restart();
    // Create input, output and debug buffers.
//    __global double ecrit,
//    __global double lrate,
//    __global double error,
//    __global double temp,
//    __global double matrix,
//    __global int NumberOfVectors,
//    __global int NumOfClasses,
//    __global int NetLength,
//    __private double ** weight,
//    __private int * mixNum,
//    __private double * output,
//    __private bool answer,
//    __private double outError,
//    __private double * outputClass,
//    __global double * NumberOfErrors,
//    __private int NumOfThread
    cl_mem params0Buf;
    cl_mem matrixBuf;
    cl_mem params1Buf;
    cl_mem weightBuf;
    cl_mem mixNumBuf;
    cl_mem outputBuf;
    cl_mem answerBuf;
    cl_mem outErrorBuf;
    cl_mem outputClassBuf;
    cl_mem numOfErrorsBuf;

    cl_mem randArrBuf;
    int bufferCounter = 0;


//    CL_INVALID_CONTEXT;
//    CL_INVALID_VALUE;
//    CL_INVALID_BUFFER_SIZE;
//    CL_INVALID_HOST_PTR;
//    CL_MEM_OBJECT_ALLOCATION_FAILURE;
//    CL_OUT_OF_RESOURCES;
//    CL_OUT_OF_HOST_MEMORY;
    cl_float *params0 = new cl_float [3];
    params0[0] = ecrit;
    params0[1] = lrate;
    params0[2] = temp;

    params0Buf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * 3,
                              params0,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    float *matrixArray = new float [global_work_size * (NetLength + 2)];
    for(int i = 0; i < global_work_size; ++i)
    {
        for(int j = 0; j < (NetLength + 2); ++j)
        {
            matrixArray[i * (NetLength + 2) + j] = matrix[i][j];
        }
    }

    matrixBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * global_work_size * (NetLength + 2),
                              matrixArray,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    cl_int *params1 = new cl_int [3];
    params1[0] = global_work_size;
    params1[1] = NumOfClasses;
    params1[2] = NetLength;
    params1Buf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_int) * 3,
                              params1,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    weightBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_float) * NumOfClasses * (NetLength + 1),
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }


    mixNumBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_int) * global_work_size,
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    outputBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_float) * NumOfClasses,
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    answerBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_int) * global_work_size,
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    outErrorBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_float) * global_work_size,
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    outputClassBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_float) * NumOfClasses,
                              NULL,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    //global:
    numOfErrorsBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_int) * NumOfClasses,
                              NumberOfErrors,
                              &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    randArrBuf = clCreateBuffer(context,
                                CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_int) * 1000,
                                randArr,
                                &clError);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer " << bufferCounter << " : " << errorMessage(clError) << endl;
        exit(clError);
    }
    else
    {
        cout << "Memory buffer " << bufferCounter++ << " created" << endl;
    }

    cout << "buffers ready, elapsed " << myTime.elapsed()/1000. << " sec" << endl;
    myTime.restart();


//    constant float * params0,
//    global float * matrix, //NumberOfVectors * (NetLength+2),
//    constant int * params1,
//    private float * weight, //NumberOfClasses * (NetLength+1)

//    local int * mixNum,
//    global float * output,

//    global int * answer,
//    global float * outError,
//    global float * NumberOfErrors,
//    constant int * randArr

    int argCounter = 0;

    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(params0Buf), (void*) &params0Buf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(matrixBuf), (void*) &matrixBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(params1Buf), (void*) &params1Buf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(weightBuf), (void*) &weightBuf);

//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(mixNumBuf), (void*) &mixNumBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(int) * global_work_size, NULL);

    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(outputBuf), (void*) &outputBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(float) * 3, NULL);

    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(answerBuf), (void*) &answerBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(outErrorBuf), (void*) &outErrorBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(numOfErrorsBuf), (void*) &numOfErrorsBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(randArrBuf), (void*) &randArrBuf);




    cout << "kernelArgs are set, elapsed " << myTime.elapsed()/1000. << " sec" << endl;
    myTime.restart();

    size_t local_work_size = 1;

    clEnqueueNDRangeKernel( queue,
                            leaveOneOutKernel,
                            1,
                            NULL,
                            &global_work_size,
                            &local_work_size,
                            0, NULL, NULL);

    clFinish( queue );



    //    values to look at the results
    cl_bool *returnedAnswer;
    cl_float *returnedError;


    returnedAnswer = (cl_bool *) clEnqueueMapBuffer( queue,
                                                  answerBuf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  sizeof(cl_bool) * global_work_size,
                                                  0, NULL, NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer: " << errorMessage(clError) << endl;
        exit(clError);
    }

    returnedError = (cl_float *) clEnqueueMapBuffer( queue,
                                                  outErrorBuf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  sizeof(cl_float) * global_work_size,
                                                  0, NULL, NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer: " << errorMessage(clError) << endl;
        exit(clError);
    }

    for(int i = 0; i < global_work_size; ++i)
    {
//        cout << "Error = " << returnedError[i] << "\tAnswer = " << returnedAnswer[i] <<endl;
    }

    bufferCounter = 0;
    clError = clEnqueueUnmapMemObject(queue,
                                      answerBuf,
                                      returnedAnswer,
                                      0,
                                      NULL,
                                      NULL);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot unmap memory buffer" << bufferCounter++ << ": " << errorMessage(clError) << endl;
        exit(clError);
    }
    clError = clEnqueueUnmapMemObject(queue,
                                      outErrorBuf,
                                      returnedError,
                                      0,
                                      NULL,
                                      NULL);
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot unmap memory buffer" << bufferCounter++ << ": " << errorMessage(clError) << endl;
        exit(clError);
    }


    cout<<"end"<<endl;

    for(int i=0; i<global_work_size; ++i)
    {
//        delete []matrix[i];
//        delete []FileName[i];
    }


//    cout<<sizeof(float) << "  " << sizeof(char) << "  " << sizeof(int) <<endl;
//    cout << (void*)kernel_source0 << endl << matrix << endl << FileName << endl << params0 << endl << params1 << endl << matrixArray << endl << randArr << endl << NumberOfErrors << endl;


//    delete []kernel_source0;
//    delete []matrixArray;
//    delete []matrix;
//    delete []FileName;
//    delete []params0;
//    delete []params1;
//    delete []randArr;
//    delete []NumberOfErrors;
    return 0;
}

