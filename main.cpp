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

const char* kernelFromFile(char* path)
{
    char* tempString = new char [200];
    char* shaderString = new char [10000];
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







int main()
{
    double ** matrix;
    int NumberOfVectors = 0;
    int NetLength = 247*19;
    int ns = 19;
    int spLength = 247;
    int NumOfClasses = 3;
    QString helpString;
    double ecrit = 0.3;
    double lrate = 0.4;
    double temp = 10.;
    double Error = 0.;
    int * NumberOfErrors;
    int * randArr = new int [1000];
    int randArr0[] = {691327682, 1726747605, 190648520, 154603954, 639873973, 93192590, 1616005123, 2123669672, 1484184336, 1125857098, 292692616, 1169317344, 1446507026, 1197937231, 1805456868, 1505064519, 942392860, 1078699843, 2038067680, 395754091, 1687359465, 1786474384, 100894089, 2084016359, 1298570611, 488906254, 1785664634, 452064338, 182992953, 1739274165, 1878652976, 874320635, 1318538122, 2069301496, 1028924590, 1958412096, 15010438, 497446065, 1934598120, 1499194775, 1623303163, 79807088, 521028471, 922326542, 1277744319, 179001691, 279907413, 72653531, 1257701535, 170491446, 468407622, 797577352, 1956965830, 569301712, 734110063, 1108052794, 1058207966, 372291049, 1560117132, 1241200919, 2111565214, 1291286460, 2115521555, 1282619689, 1213104308, 996962497, 1093548137, 1228114747, 1494408562, 880662609, 579825874, 970228077, 960469697, 1100854345, 1892554619, 90730368, 1279856036, 24978385, 163383899, 390073923, 195469831, 631791521, 1187651275, 4952013, 1201093233, 1921761338, 1113004807, 111817551, 146568739, 525638291, 1353018471, 110650306, 1816924751, 1321056378, 1393269995, 882545412, 170535227, 339334484, 2110660159, 1664943789, 1219997093, 543002385, 487688218, 32983142, 1643856730, 232759190, 123713510, 776229118, 257737575, 287097409, 1166303042, 453207406, 918888930, 206470669, 458159419, 2119982164, 2128232008, 1571164227, 84316067, 127317099, 2096802518, 1437334538, 237967405, 1766243622, 610907268, 1631237400, 501305386, 781442495, 1970571884, 464481897, 298902636, 1043085329, 1007484282, 786590855, 1076068471, 503857364, 1019350045, 1199781981, 1280086482, 1277087620, 1486879390, 298905876, 1730295026, 258284673, 505376546, 40970797, 230783189, 486124906, 1612135024, 315099256, 613442005, 1561453895, 1752433795, 851409411, 1180213869, 215857415, 335163163, 1681519255, 997299911, 158251400, 2146001152, 1296202547, 1201336729, 1006001786, 2082793402, 129921553, 1509859150, 954659799, 1329703534, 642461984, 84263771, 669099277, 941367861, 1814558797, 927383950, 1446744407, 1855529595, 1158167139, 1932869313, 1320180971, 1473266395, 398827670, 734151218, 1078216542, 1250237081, 1914365087, 1294073958, 1585400245, 1448400694, 143890221, 1743651645, 1446918198, 1440092768, 797504726, 305436336, 1375402523, 927426279, 1815295486, 182578674, 109646166, 310273823, 266842446, 778745443, 1251641684, 2081401243, 1706129393, 550902443, 1789447190, 716812884, 336288108, 962144514, 42595631, 735115778, 1696295732, 1120812174, 1985352860, 1463177172, 267402484, 1423269457, 764094218, 411292705, 1019437454, 63528769, 1851385473, 1816942180, 368965105, 1079304348, 596884812, 36776944, 1261883023, 706530978, 347050767, 1528725469, 1485276421, 1598692451, 1462643064, 1043922166, 2111246, 1104606607, 1760735050, 338399354, 2066751121, 1803330681, 1073515132, 1615563205, 776659207, 911384344, 931256729, 1044061691, 187170153, 1695350948, 1455354396, 1206607607, 1758879717, 1159256222, 876066140, 2127844822, 91076922, 1472950952, 17138118, 1352959945, 31998282, 364188885, 734201766, 1517274703, 1962881336, 49361183, 413713221, 1964992582, 1153967790, 26964623, 155908288, 1073235263, 1830295304, 1229423421, 541314820, 459470864, 2140807765, 1472571550, 1503532555, 180494271, 1020438850, 811403304, 1387101878, 631834919, 1970659526, 115684370, 612196093, 2061736448, 1588635322, 629334212, 1267212746, 1620633604, 993523097, 2001414512, 990424659, 808920786, 2050775695, 1404137880, 626429720, 1057259837, 1431102503, 782338009, 2130495100, 1113914160, 2011761430, 524326273, 1573385024, 2005085547, 1996897823, 929433931, 38096170, 869853025, 1740837235, 1425198049, 1501687944, 1564013113, 1540882419, 2113884037, 1478265914, 982034094, 595734601, 597995012, 455184050, 1589257699, 451925876, 1445608710, 250694837, 355217924, 702262942, 877124557, 1412477761, 2133365446, 1659462566, 1395489214, 1099795958, 1523740348, 1919815487, 525697334, 1381342248, 1769229662, 1455131265, 1419438418, 491599039, 1048484853, 697152819, 1993286983, 465014318, 90551591, 1959687372, 1943280232, 1072585685, 407938326, 393791596, 1527769735, 1997196025, 845717473,
    825894797, 100407214, 1200935397, 1528157740, 977531771, 465929510, 1514039538, 489510690, 1861418724, 466351848, 2013251038, 1633750563, 992049182, 1247109638, 1255496577, 299696799, 519064409, 1747095616, 1348181652, 1216217228, 1592898951, 1813195971, 1306768819, 1405102676, 1608992555, 231870856, 1813041002, 2002784152, 1759640592, 1662753379, 701017977, 438051741, 1763160593, 1901953374, 1966209481, 593208716, 220399236, 1332765371, 1082719406, 2081817961, 1799117219, 948486797, 1568084876, 643682753, 48112787, 676097806, 943379553, 567177196, 275709774, 144077557, 1783394425, 1868608726, 1957273528, 942679596, 1126227754, 1418782436, 1174550453, 791785108, 1274082940, 786707397, 307054839, 1975100917, 1224759138, 2070215432, 1729570643, 1043484972, 515940500, 1949969879, 228766695, 1598659907, 1884304192, 2027883915, 399663056, 1304905421, 524083020, 447775843, 1981003227, 1467462573, 1014953040, 109229353, 1611540131, 650863817, 1977838079, 1421330011, 1593543413, 956582185, 692628799, 620610218, 1748367293, 1966711739, 1407317615, 2055422132, 1794329008, 484593106, 1978153916, 1376416003, 1528078078, 346610769, 1178902235, 1756844773, 1945270676, 915722779, 1637245040, 197450084, 73144552, 13844413, 645225927, 2054147779, 1481306986, 1660178967, 15893485, 945363469, 163559136, 1993731564, 219209833, 1757102550, 802830102, 911838632, 230229120, 403713747, 731066724, 1637546736, 311652232, 377912084, 2122139842, 142322500, 1754328088, 1502734272, 488933269, 785746675, 1112095397, 286720297, 1701469454, 601856790, 484170381, 1774614007, 615701203, 1129396309, 1681278138, 2097008189, 642091628, 1697171623, 894888011, 805650765, 1543419540, 1114097844, 415269667, 198765994, 2025936476, 645498787, 602479741, 609519552, 135561875, 914131973, 987431637, 110218069, 1056454474, 594276077, 1612952341, 1545387743, 1380022752, 577564091, 1832108041, 934008558, 1179420881, 168794774, 561138917, 1795122084, 1298191083, 94933408, 1744646625, 1940282712, 1792105031, 492050988, 598449829, 1188040923, 1606148832, 1013719496, 1386806917, 1484601661, 1659218283, 1989286659, 2094121213, 1794780159, 755934984, 934069202, 1904998228, 1812389458, 1528345279, 1370466922, 1210293554, 760884383, 1948031013, 894917947, 1694892942, 979968246, 1063712721, 108548211, 627606682, 214420157, 203481619, 224769659, 7219221, 1995586651, 716820648, 605669050, 1036143926, 175485832, 1619388546, 275467196, 1660087493, 1131123181, 117270207, 1606725059, 778419692, 873205191, 393310613, 535934273, 538111002, 1921655893, 1906401195, 1748404556, 535056628, 1706948560, 495838855, 82465922, 539433158, 1559551576, 191014134, 1167039840, 1773971733, 394495753, 1391809499, 1781190954, 242598756, 2108630147, 239376356, 1278742683, 136632332, 1858764902, 1554209879, 1796719825, 842404436, 1671480086, 1255961236, 1620824128, 397201629, 1649271850, 9274753, 935312631, 1423444095, 1915675948, 536233539, 1958500723, 1475140860, 1032072394, 2040966646, 2014574018, 444140323, 84497132, 1034130210, 70628408, 478992885, 278456062, 1851819363, 721591642, 239602561, 2091195719, 2000334325, 376234893, 1802476974, 1407060556, 25471071, 497397762, 931056994, 1281432307, 2118221890, 1328258623, 783220509, 2127496644, 116087607, 59180956, 1895688944, 652321146, 2017681680, 1223346157, 1684393541, 1911164678, 1090436527, 2128533864, 1995661810, 2124566738, 51678624, 327171047, 255539152, 1903497987, 1048762689, 495141713, 1847210059, 901613366, 871376607, 1502203385, 161190274, 896847678, 1999601147, 1092247268, 30796337, 1970339389, 273022244, 814016847, 1950352385, 389109851, 873197803, 1698557682, 1041430997, 743395835, 774420191, 578340890, 507076865, 1864856718, 559391106, 355255027, 1841939808, 611069731, 682426075, 2097478960, 367084070, 1731188764, 445137026, 66810481, 485318483, 1316513633, 1569013866, 646508757, 65877663, 1421131365, 1738756026, 96674000, 1243987107, 2011778270, 910690847, 1046855844, 253404473, 1783888651, 597929878, 1294835470, 379800838, 1372350069, 1873176361, 886877704, 1089723140, 285083819, 1242132731, 784179300, 896153550, 1924558806, 734174613, 1263237621, 1508263923, 1179311639, 1330048102, 1993582406, 348341624, 751578321, 492607515, 414219287, 25226038, 83879893, 510893287, 1269213145, 2095658163, 1421584135, 168585342, 201578988, 1057989138, 766515220, 1496414459, 1437789976, 2138865290, 1222107172, 177184032, 1081104782, 1507190991, 1419316764, 1865284082, 255860894, 1196391922, 451975047, 1519098515, 557172197, 1631286686, 701662969, 403270955, 1979628310, 1453241290, 895878471, 246363949, 1478467329, 979758364, 757257237, 600196826, 927932880, 31357724, 768782168, 1129511868, 1089346862, 1535297389, 478442679, 379653190, 1526679031, 1700549851, 556837223, 460300165, 1060257195, 1976153987, 178100599, 1316118089, 1025062261, 630075647, 687732956, 1582234459, 113878685, 1389395925, 1985505414, 2093506996, 695153568, 733900237, 192387297, 26137249, 1713658602, 949644534, 626334075, 494107834, 981002258, 1395116244, 1623619702, 2070349120, 782929985, 2102062382, 302518663, 162125368, 1655128585, 859355886, 622425533, 567902132, 688026225, 800526132, 1884020221, 1713088486, 1430601779, 424269529, 1147839297, 1544480465, 1813665455, 985861064, 1490503813, 361335375, 1719761301, 1682891110, 387472624, 1285936255, 485051997, 1013806699, 1780044089, 1466054255, 261439295, 1256180144, 1388919728, 1044369280, 1210758878, 1691438391, 1206494648, 718403815, 403310629, 1828920181, 1286305948, 1091336854, 481962666, 1022842521, 656941692, 1912564445, 1447112051, 1804780990, 1309561262, 1113293858, 643158406, 652581427, 1474629233, 215436059, 187988890, 1862101857, 1501372315, 673040887, 728424908, 1133932756, 2139095142, 989864204, 242629252, 1380531222, 2034233484, 1453388130, 924485965, 1093244485, 24308298, 1327796594, 774681018, 1310614246, 271649800, 1256643684, 185973119, 928591493, 1021724482, 1633085170, 585888835, 183802096, 598895380, 1229047241, 836383524, 2073524613, 1444483300, 1024372414, 1788142822, 798371967, 1697413301, 369084083, 1932304724, 1689024795, 1358948287, 27450328, 922072370, 1245698123, 1480838459, 1846558335, 191458960, 1505146757, 1026871282, 966139979, 668277355, 1298521082, 75300015, 854250474, 79628927, 1097024497, 339851997, 665517762, 1280826594, 938747377, 1894565003, 2117210118, 864788343, 1191564656, 994098884, 505447517, 1989936623, 544028537, 874531600, 1774757699, 85569684, 85996239, 1802208028, 1007642054, 1331694363, 1135562839, 706716742, 1523153323, 493225948, 1733588024, 341809654, 1161503303, 884625458, 417109670, 2015753777, 964254386, 1514134167, 208122126, 1629772148, 647477113, 1146869504, 1376853504, 617203583, 2011657847, 420934512, 1611302467, 369621716, 263387487, 7847356, 1244153317, 2038145187, 93417041, 1330149556, 1692869567, 1101059095, 514360271, 680948758, 1807775837, 2037513595, 1174174706, 1393880213, 231839601, 188194361, 131022024, 648949271, 56464490, 1095276410, 15599791, 264586617, 577564910, 663076904, 1411456121, 1954418414, 1280280488, 1275630320, 227869278, 744099307, 1645252036, 491256766, 751946664, 741921705, 381918305, 845363705, 2072071262, 2074787872, 1946422800, 438947885, 608252982, 1606714990, 328977832, 1782427688, 853111555, 560817434, 1970622049, 984133579, 1209766705, 2027086539, 2079409989, 1225366496, 144189508, 509491252, 1888443401, 1555645629, 316426018, 1021240241, 683792301};
    for(int i = 0; i < 1000; ++i)
    {
        randArr[i] = randArr0[i];
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
    const char *kernel_source = (const char*)kernelFromFile("/home/michael/Qt/Projects/myOpenCL/kernel.cl"); //generality
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
    clError = clBuildProgram( program, 1, &device, NULL, NULL, NULL);
    cout << "end build program" << endl;
    //Tell compiler to dump intermediate .il and .isa GPU files.
    // 5. Print compiler error messages
    if(clError != CL_SUCCESS)
    {
        cout << "clBuildProgram failed: " << errorMessage(clError) << endl;
        char buf[0x10000];
        clGetProgramBuildInfo( program,
                               device,
                               CL_PROGRAM_BUILD_LOG,
                               0x10000,
                               buf,
                               NULL);
        cout << buf << endl;
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
    cl_mem ecritBuf;
    cl_mem lrateBuf;
    cl_mem errorBuf;
    cl_mem tempBuf;
    cl_mem matrixBuf;
    cl_mem params1Buf;
    cl_mem numOfVectsBuf;
    cl_mem numOfClassesBuf;
    cl_mem netLengthBuf;
    cl_mem weightBuf;
    cl_mem mixNumBuf;
    cl_mem outputBuf;
    cl_mem answerBuf;
    cl_mem outErrorBuf;
    cl_mem outputClassBuf;
    cl_mem numOfErrorsBuf;
    cl_mem NumOfThreadBuf;
    cl_mem NumOfVectorToSkipBuf;
    cl_mem randArrBuf;
    int bufferCounter = 0;


//    CL_INVALID_CONTEXT;
//    CL_INVALID_VALUE;
//    CL_INVALID_BUFFER_SIZE;
//    CL_INVALID_HOST_PTR;
//    CL_MEM_OBJECT_ALLOCATION_FAILURE;
//    CL_OUT_OF_RESOURCES;
//    CL_OUT_OF_HOST_MEMORY;
    double *params0 = new double [3];
    params0[0] = ecrit;
    params0[1] = lrate;
    params0[2] = temp;

    params0Buf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double) * 3,
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


    ecritBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double),
                              &ecrit,
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

    lrateBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double),
                              &lrate,
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

    errorBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double),
                              &Error,
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

    tempBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double),
                              &temp,
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

    double *matrixArray = new double [NumberOfVectors * (NetLength + 2)];
    for(int i = 0; i < NumberOfVectors; ++i)
    {
        for(int j = 0; j < (NetLength + 2); ++j)
        {
            matrixArray[i * (NetLength + 2) + j] = matrix[i][j];
        }
    }

    matrixBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_double) * NumberOfVectors * (NetLength + 2),
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

    int *params1 = new int [3];
    params1[0] = NumberOfVectors;
    params1[1] = NumOfClasses;
    cout<<NumOfClasses;
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


    numOfVectsBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_int),
                              &NumberOfVectors,
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

    numOfClassesBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_int),
                              &NumOfClasses,
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

    netLengthBuf = clCreateBuffer(context,
                              CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_int),
                              &NetLength,
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

    //privates:

    weightBuf = clCreateBuffer(context,
                              CL_MEM_READ_WRITE,
                              sizeof(cl_double) * NumOfClasses * (NetLength + 1),
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
                              sizeof(cl_double) * NumOfClasses,
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
                              sizeof(cl_double) * global_work_size,
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
                              sizeof(cl_double) * NumOfClasses,
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
                              &NumberOfErrors,
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

    //private:
    NumOfThreadBuf = clCreateBuffer(context,
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

    NumOfVectorToSkipBuf = clCreateBuffer(context,
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

//    cout<<bufferCounter<<endl;
    randArrBuf = clCreateBuffer(context,
                                CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_int) * 100,
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



    int argCounter = 0;
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(params0Buf), (void*) &params0Buf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(ecritBuf), (void*) &ecritBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(lrateBuf), (void*) &lrateBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(errorBuf), (void*) &errorBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(tempBuf), (void*) &tempBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(matrixBuf), (void*) &matrixBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(params1Buf), (void*) &params1Buf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(numOfVectsBuf), (void*) &numOfVectsBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(numOfClassesBuf), (void*) &numOfClassesBuf);
//    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(netLengthBuf), (void*) &netLengthBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(weightBuf), (void*) &weightBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(mixNumBuf), (void*) &mixNumBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(outputBuf), (void*) &outputBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(answerBuf), (void*) &answerBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(outErrorBuf), (void*) &outErrorBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(outputClassBuf), (void*) &outputClassBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(numOfErrorsBuf), (void*) &numOfErrorsBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(NumOfThreadBuf), (void*) &NumOfThreadBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(NumOfVectorToSkipBuf), (void*) &NumOfVectorToSkipBuf);
    clSetKernelArg(leaveOneOutKernel, argCounter++, sizeof(randArrBuf), (void*) &randArrBuf);


    cout << "kernelArgs are set, elapsed " << myTime.elapsed()/1000. << " sec" << endl;
    myTime.restart();

    clEnqueueNDRangeKernel( queue,
                            leaveOneOutKernel,
                            1,
                            NULL,
                            &global_work_size,
                            NULL,
                            0, NULL, NULL);

    clFinish( queue );



    //    values to look at the results
    cl_bool *returnedAnswer;
    cl_double *returnedError;
    cl_int *returnedNumofthread;
    cl_int *returnedNumOfSkipped;


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

    returnedError = (cl_double *) clEnqueueMapBuffer( queue,
                                                  outErrorBuf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  sizeof(cl_double) * global_work_size,
                                                  0, NULL, NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer: " << errorMessage(clError) << endl;
        exit(clError);
    }

    returnedNumofthread = (cl_int *) clEnqueueMapBuffer( queue,
                                                  NumOfThreadBuf,
                                                  CL_TRUE,
                                                  CL_MAP_READ,
                                                  0,
                                                  sizeof(cl_int) * global_work_size,
                                                  0, NULL, NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer: " << errorMessage(clError) << endl;
        exit(clError);
    }

    returnedNumOfSkipped = (cl_int *) clEnqueueMapBuffer( queue,
                                                         NumOfVectorToSkipBuf,
                                                         CL_TRUE,
                                                         CL_MAP_READ,
                                                         0,
                                                         sizeof(cl_int) * global_work_size,
                                                         0, NULL, NULL, &clError );
    if (clError != CL_SUCCESS)
    {
        cout << "Cannot create memory buffer: " << errorMessage(clError) << endl;
        exit(clError);
    }

    for(int i = 0; i < global_work_size; ++i)
    {
        cout << "NumOfThread = " << returnedNumofthread[i] << " NumOfSkipped = " << returnedNumOfSkipped[i] << "\tError = " << returnedError[i] << "\tAnswer = " << returnedAnswer[i] <<endl;
//        cout << "NumOfThread = " << returnedNumofthread[i] <<endl;
    }


//    delete []params0;
//    delete []params1;
//    delete []NumberOfErrors;
//    delete []matrixArray;
    return 0;
}

