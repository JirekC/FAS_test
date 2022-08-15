#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "FAS.cl/fas.hpp"
#include <nlohmann/json.hpp> // JSON parser for project files
using json = nlohmann::json;

template<typename T>
void LoadF32FromFile(std::ifstream &data_file, std::vector<T> &mem)
{
    size_t first_data_byte = data_file.tellg(); // remember actual position (probably begin of file)
    data_file.seekg(0, data_file.end);
    size_t length = data_file.tellg();
    length -= first_data_byte;
    data_file.seekg(first_data_byte, data_file.beg);
    mem.resize(length / sizeof(T));
    data_file.read((char*)mem.data(), sizeof(T) * (length / sizeof(T))); // round to lower integer multiply of sizeof(T)
}

template<typename T>
fas::vec3<T> parse_vec3(json jvec3)
{
    fas::vec3<T> ret_vec;
    json val;

    if(!jvec3.is_object())
    {
        throw std::runtime_error("3D vector not found");
    }
    if(!(val = jvec3["x"]).is_number())
    {
        throw std::runtime_error("invalid 3D vector format");
    }
    ret_vec.x = val;
    if(!(val = jvec3["y"]).is_number())
    {
        throw std::runtime_error("invalid 3D vector format");
    }
    ret_vec.y = val;
    if(!(val = jvec3["z"]).is_number())
    {
        throw std::runtime_error("invalid 3D vector format");
    }
    ret_vec.z = val;

    return ret_vec;
}

template<typename T>
fas::vec2<T> parse_vec2(json jvec2)
{
    fas::vec2<T> ret_vec;
    json val;

    if(!jvec2.is_object())
    {
        throw std::runtime_error("3D vector not found");
    }
    if(!(val = jvec2["x"]).is_number())
    {
        throw std::runtime_error("invalid 3D vector format");
    }
    ret_vec.x = val;
    if(!(val = jvec2["y"]).is_number())
    {
        throw std::runtime_error("invalid 3D vector format");
    }
    ret_vec.y = val;

    return ret_vec;
}

int main(int argc, char* argv[])
{
    fas::device* dev; // TODO: allow use of multiple devices - each field can select GPU ...
    int sim_steps = 10e3; // total # of simulation steps
    fas::data_t dt; // time-step [sec]
    fas::data_t dx; // space-step [m]
    std::vector<fas::material> materials; // all used materials - common for whole project
    std::vector<fas::field*> fields; // simulation fields - TODO: connect together by boundaries ...
    fas::driver* drv; // TODO: predelat na vector
    std::vector<fas::scanner*> scanners; // scanners from all fields together

    if (argc < 2)
    {
        std::cout << "ERR: no input file.\n";
        std::cout << "Using: FAS [project_file.json]\n";
        exit(-1);
    }

    /**********************************************************/
    /*** print platforms & devices available in this system ***/
    /**********************************************************/
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> devices;
    // print platforms & devices info
    for (int i = 0; i < platforms.size(); i++) {
        std::cout << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (int j = 0; j < devices.size(); j++) {
            std::cout << "\t" << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }
    std::cout << std::endl;

    /*******************************************************/
    /*** select platform & device calculation to be done ***/
    /*******************************************************/
    int plat_idx, dev_idx;
    std::cout << "Select platform (index): ";
    std::cin >> plat_idx;
    if (plat_idx < 0 || plat_idx >= platforms.size()) {
        std::cout << "ERR: not valid platform index\n";
        return -1;
    }
    platforms[plat_idx].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cout << "Select device from platform (index): ";
    std::cin >> dev_idx;
    if (dev_idx < 0 || dev_idx >= devices.size()) {
        std::wcout << "ERR: not valid device index\n";
        return -1;
    }

    /********************************************************/
    /*** prepare device : compile program, create kernels ***/
    /********************************************************/
    try {
        dev = new fas::device(devices[0], "../lib/FAS.cl/fas.cl");
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    /************************************/
    /*** load project from .json file ***/
    /************************************/
    try
    {
        int cntr;
        json jproject, val;
        std::ifstream project_file;
        project_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // try to open and parse input project-file
        project_file.open(argv[1]);
        jproject = json::parse(project_file);

        // overal project parameters
        val = jproject["steps"];
        if(!val.is_number_unsigned()) throw std::runtime_error("A valid number of simulation steps was not specified");
        sim_steps = val;
        if(!(val = jproject["dt"]).is_number()) throw std::runtime_error("A valid time-step (dt) was not specified");
        dt = val;
        if(!(val = jproject["dx"]).is_number()) throw std::runtime_error("A valid space-step (dx) was not specified");
        dx = val;

        // show some summary
        std::cout << std::endl << "Simulation summary:" << std::endl;
        std::cout << "Maximum frequency (based on time-step): " << (0.5f / dt) << " Hz" << std::endl;
        std::cout << "Loading materials (Courant nr best bellow 0.577):" << std::endl;

        // load & recalculate all materials used in project
        for(auto jm : jproject["materials"])
        {
            fas::material m;
            m.c = jm["c"]; // TODO: try : catch
            m.ro = jm["ro"];
            m.name = jm["name"];
            fas::data_t lambda = m.c * dt / dx;
            fas::data_t fmax = 0.5f * m.c / dx; // Shannon-Kotelnik
            std::cout.precision(3);
            std::cout << "Material \"" << m.name << "\": \tCourant nr = " << lambda << "\tf_max = " << fmax << " Hz, \tcharacteristic impedance = " << m.c * m.ro << std::endl;
            materials.push_back(m);
        }
        if(materials.size() == 0)
            throw std::runtime_error("No material specified, specify at least one");
        fas::material::Recalc(materials);

        // load fields
        for(auto jf : jproject["fields"])
        {
            std::cout << "Creating new field: ";
            if((val = jf["name"]).is_string())
                std::cout << std::string(val);
            std::cout << "\n";
            fas::vec3<uint32_t> size;
            try
            {
                size = parse_vec3<uint32_t>(jf["size"]);
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: size is not specified:\n" + e.what());
            }

            // init field
            fas::field* f = new fas::field(*dev, size, dx, dt);
            f->materials = materials; // copy materials to field's private memory
            if((val = jf["calc_rms"]).is_boolean())
            {
                f->Prepare(val);
                if(val == true)
                {
                    // load rms window from "rms_window_file"
                    std::ifstream rms_file;
                    rms_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                    try
                    {
                        rms_file.open(jf["rms_window_file"]);
                        LoadF32FromFile(rms_file, f->rms_window);
                        rms_file.close();
                    }
                    catch(const std::exception& e)
                    {
                        throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: loading \"rms_window_file\":\n" + \
                                                std::string(e.what()) + "\nCheck file-name");
                    }
                }
            }
            else
            {
                f->Prepare(false); // no rms calculation by default
            }
            f->Clear();

            // TODO: drivers




    /******************************/
    /*** create circular driver ***/
    /******************************/
    std::cout << "Creating driver(s).\n";
    try {
        fas::object::CreateCylinder(*f, { 380,150,128 }, { 0, M_PI* 0.5, M_PI * 0.4/*72°*/}, {30,30,1}, 0x80); // 0x80 -> material.MSB is set, so object is transducer

        drv = new fas::driver(*f);
        drv->CollectElements(); // CollectElements() will clear MSBs of all elements after collection complete
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }
    // write driver's elements (uint64_t num of elements, element's coordinates - format: { x0, x1 ... xn, y1 .. yn, z1 .. zn }
    FILE* driverf = fopen("driver", "wb"); // driver de
    uint64_t tmp_ne = drv->num_elements; // one can merge more than one driver into one file
    fwrite(&tmp_ne, sizeof(uint64_t), 1, driverf);
    fwrite(drv->GetElementsCoords().data(), sizeof(uint32_t), drv->num_elements * 3, driverf);
    fclose(driverf);





            
            // create scanners
            cntr = 0;
            for( auto jscan : jf["scanners"] )
            {
                fas::vec3<uint32_t> position;
                fas::vec2<uint32_t> size;
                fas::vec3<double> rotation;
                std::string file_name;
                uint32_t store_every_nth_frame = 1;

                // parse position, size and rotation
                try
                {
                    position = parse_vec3<uint32_t>(jscan["position"]);
                    size = parse_vec2<uint32_t>(jscan["size"]);
                    rotation = parse_vec3<double>(jscan["rotation"]);
                }
                catch(const std::exception& e)
                {
                    throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: " \
                        "scanner [" + std::to_string(cntr) + "]: position, size or rotation vector has invalid format");
                }
                // try to parse file_name
                if((val = jscan["out_file"]).is_string())
                {
                    file_name = val;
                }
                else
                {
                    // no file name specified, use some default
                    file_name = "f" + std::to_string(fields.size()) + "s" + std::to_string(cntr) + "data.f32";
                }
                // try to parse how many frames to store (default is 1 - every frame)
                if((val = jscan["store_every_nth_frame"]).is_number_unsigned())
                {
                    store_every_nth_frame = val;
                    if(store_every_nth_frame < 1)
                        store_every_nth_frame = 1;
                }

                fas::scanner* scan = new fas::scanner(*f, position, rotation, size, file_name, store_every_nth_frame);
                scanners.push_back(scan);
                cntr++;
            }

            // create objects inside field
            f->Clear(); // restart all material_ids to #0
            cntr = 0;
            for( auto jo : jf["objects"] )
            {
                fas::vec3<uint32_t> position;
                fas::vec3<uint32_t> size;
                fas::vec3<double> rotation;
                uint mat_id;

                // parse position, size and rotation
                try
                {
                    position = parse_vec3<uint32_t>(jo["position"]);
                    size = parse_vec3<uint32_t>(jo["size"]);
                    rotation = parse_vec3<double>(jo["rotation"]);
                }
                catch(const std::exception& e)
                {
                    throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: " \
                        "object [" + std::to_string(cntr) + "]: position, size or rotation vector has invalid format");
                }

                // parse material id
                if(!(val = jo["material_id"]).is_number_unsigned())
                {
                    std::cout << "Field [" + std::to_string(fields.size()) + "]: " \
                        "object [" + std::to_string(cntr) + "]: \"material_id\" not specified - assuming #0\n";
                    mat_id = 0;
                }
                else
                {
                    mat_id = val;
                    if(mat_id > 255)
                    {
                        std::cout << "Field [" + std::to_string(fields.size()) + "]: " \
                            "object [" + std::to_string(cntr) + "]: \"material_id\" greater than 255 - #0 will be used instead\n";
                        mat_id = 0;
                    }
                }

                // parse shape & create object, all shaped are 3D and filled
                if(!(val = jo["shape"]).is_string())
                {
                    throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: object [" + std::to_string(cntr) + "]: shape not specified");
                }
                if(val == "box")
                {
                    fas::object::CreateBox(*f, position, rotation, size, mat_id);
                }
                else if(val == "cylinder")
                {
                    fas::object::CreateCylinder(*f, position, rotation, size, mat_id);
                }
                else if(val == "ellipsoid")
                {
                    fas::object::CreateEllipsoid(*f, position, rotation, size, mat_id);
                }
                else
                {
                    throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: object [" + std::to_string(cntr) + "]: unknown shape");
                }

                cntr++;
            }
            f->cl_queue.finish(); // wait for all work done (on device side)
            fields.push_back(f);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "ERR: Preparing scene from file \"" << argv[1] << "\": " << e.what() << '\n';
        exit(-1);
    }
    

    
    fas::data_t freq = 5e3; // [Hz]

    /*************************************/
    /*** define rms integration window ***/
    /*************************************/
    // fas::data_t TN = 1.0 / freq / dt; // one period of signal in steps
    // f->rms_window.resize(sim_steps);
    // for (int i = 0; i < sim_steps; i++) {
    //     f->rms_window[i] = i < sim_steps - TN ? 0.0 : 1.0; // rectangular window
    // }

    


    /****************************************/
    /*** run simulation - kernels in loop ***/
    /****************************************/
    std::cout << "Starting simulation ...\n";
    fas::data_t time = 0.0;
    for (int i = 0; i < sim_steps; i++, time += dt) {
        printf("\rsimulation step: %i/%i (%f sec)", i + 1, sim_steps, time);

        try {
            // all drivers here
            float signal;
            signal = cos(2 * M_PI * time * freq);

            drv->Drive(signal);
            // ...
            drv->f->cl_queue.enqueueBarrierWithWaitList();

            // all scanners here
            for(auto s : scanners)
            {
                s->Scan2devmem();
            }
            for(auto f : fields)
            {
                f->cl_queue.enqueueBarrierWithWaitList();
            }
            for(auto s : scanners)
            {
                s->Scan2file();
            }

            // launch sim. kernel for field(s)
            for(auto f : fields)
            {
                f->SimStep();
            }
            // Finish all work before new iteration
            for(auto f : fields)
            {
                f->Finish();
            }
        }
        catch (std::exception& e) {
            std::cout << e.what() << "\nin simulation step: " << i << "\n";
            return -1; // system will clean up everything
        }
    }
    std::cout << std::endl;

    /******************************************/
    /*** Calculation of RMS in each element ***/
    /******************************************/
    // if (calc_rms) {
    //     std::cout << "Calculate RMS.\n";
    //     try {
    //         // all fields with RMS here
    //         f->FinishRms();
    //         // ...
    //         f->Finish();
    //     }
    //     catch (std::string& text) {
    //         std::cout << text << "\n";
    //         return -1; // system will clean up everything
    //     }

    //     // store 3D array of RMS
    //     fas::data_t* rms = f->Map_rms_read();
    //     FILE* rmsf = fopen("rms0", "wb");
    //     // write header
    //     fwrite(&(f->size.x), sizeof(uint32_t), 1, rmsf);
    //     fwrite(&(f->size.y), sizeof(uint32_t), 1, rmsf);
    //     fwrite(&(f->size.z), sizeof(uint32_t), 1, rmsf);
    //     fwrite(rms, sizeof(fas::data_t), (size_t)f->size.x * f->size.y * f->size.z, rmsf);
    //     fclose(rmsf);
    // }

    delete drv;
    delete dev;

    // call destructors
    // delete[] scanners.data();
    // delete[] fields.data();

    std::cout << "Finished.\n";
    return 0;
}



/*
* F.A.S.cl example program - box with one driver
*/
/*
int main() {

    int sim_steps = 20000;
    fas::data_t dt = 0.5e-6; // [sec]
    fas::data_t dx = 5e-3; // [m]
    fas::data_t freq = 5e3; // [Hz]

    // platforms & devices available in this system
    std::vector<cl::Platform> platforms;
    cl::Platform::get( &platforms );
    // print platforms & devices info
    for( auto &p : platforms ) {
        std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices) {
            std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << std::endl;
            auto v = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_SIZES:" << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
         
        }
    }
    std::cout << std::endl;
    
    // TODO select plat. / dev.
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // prepare device: compile program, create kernels
    // prepare field: alloc memory on device, clear
    fas::device* dev;
    fas::field* f;
    try {
        dev = new fas::device(devices[0], "c:/lib/FAS.cl/fas.cl"); // TODO: copy fas.cl to same dir as executable and use relative path
        f = new fas::field(*dev, { 500, 500, 1024 }, dx, dt);
        f->materials.push_back({ 340.0, 12.256e-1, 0, 0, 0, 0 }); // Air
        f->materials.push_back({ 970.0, 0.179, 0, 0, 0, 0 }); // He
        f->materials.push_back({ 6400.0, 2700.0, 0, 0, 0, 0 }); // Al
        fas::material::Recalc(*f);
        f->Prepare(true);
        f->Clear();
    }
    catch (std::string& text) {
        std::cout << text << std::endl;
        return -1; // авария :(
    }

    // define rms integration window
    fas::data_t TN = 1.0 / freq / dt; // one period of signal in steps
    f->rms_window.resize(sim_steps);
    for (int i = 0; i < sim_steps; i++) {
        f->rms_window[i] = i < sim_steps - TN ? 0.0 : 1.0;
    }

    // create rectangle driver
    std::cout << "Create driver(s).\n";
    fas::driver* drv;
    try {
        //fas::object::CreateRect(*f, { 60,60,50 }, { 0, 0, 0 }, { 80,80 }, 0x80); // 0x80 -> material.MSB is set, so object is transducer
        fas::object::CreateCylinder(*f, { 250,250,100 }, { 0, 0, 0 }, { 20,20,1 }, 0x80); // 0x80 -> material.MSB is set, so object is transducer

        drv = new fas::driver(*f);
        drv->CollectElements(); // CollectElements() will clear MSBs of all elements ofter collection complete
    }
    catch (std::string& text) {
        std::cout << text << std::endl;
        return -1; // авария :(
    }
    // write driver's elements (uint64_t num of elements, element's coordinates - format: { x0, x1 ... xn, y1 .. yn, z1 .. zn }
    FILE* driverf = fopen("driver", "wb");
    uint64_t tmp_ne = drv->num_elements; // one can merge more than one driver into one file
    fwrite(&tmp_ne, sizeof(uint64_t), 1, driverf);
    fwrite(drv->GetElementsCoords().data(), sizeof(uint32_t), drv->num_elements * 3, driverf);
    fclose(driverf);

    // create scanner
    std::cout << "Create scanner(s).\n";
    fas::scanner* scan;
    try {
        fas::object::CreateRect(*f, { 250,0,0 }, { 0,-0.5 * M_PI,0 }, { 1024,512 }, 0x80);
        fas::object::CreateRect(*f, { 0,250,0 }, { 0, 0, 0.5 * M_PI }, { 512,1024 }, 0x80);
        scan = new fas::scanner(*f);
        scan->CollectElements();
    }
    catch (std::string& text) {
        std::cout << text << std::endl;
        return -1; // авария :(
    }

    // define some objects from different material here
    std::cout << "Create other objects(s).\n";
    fas::object::CreateBox(*f, { 150,220,100 }, { 0,0,0 }, { 200,60,1 }, 0x02);
    fas::object::CreateCylinder(*f, { 250,250,100 }, { 0, 0, 0 }, { 21,21,1 }, 0x00);
    fas::object::CreateBox(*f, { 150,220,50 }, { 0,0,0 }, { 200,60,1 }, 0x02);
    fas::object::CreateBox(*f, { 150,220,50 }, { 0,0,0 }, { 200,1,50 }, 0x02);
    fas::object::CreateBox(*f, { 150,220,50 }, { 0,0,0 }, { 1,60,50 }, 0x02);
    fas::object::CreateBox(*f, { 350,220,50 }, { 0,0,0 }, { 1,60,50 }, 0x02);
    fas::object::CreateBox(*f, { 150,280,50 }, { 0,0,0 }, { 200,1,50 }, 0x02);

    // open storage for writing simulation outputs (pressure in each element of scanner)
    FILE* dataf = fopen("data0", "wb");
    // write header (uint64_t num of elements, element's coordinates - format: { x0, x1 ... xn, y1 .. yn, z1 .. zn }
    tmp_ne = scan->num_elements;
    fwrite(&tmp_ne, sizeof(uint64_t), 1, dataf);
    fwrite(scan->GetElementsCoords().data(), sizeof(uint32_t), scan->num_elements * 3, dataf);

    // run kernels in loop
    fas::data_t time = 0.0;
    for (int i = 0; i < sim_steps; i++, time += f->dt) {
        printf("\rsimulation step: %i/%i (%f sec)", i + 1, sim_steps, time);

        // all drivers here
        drv->Drive(1.0 * sin(2 * M_PI * time * freq) );
        // ...
        drv->f->Finish();

        // all scanners here
        scan->Scan2devmem();
        // ...
        scan->f->Finish();
        scan->Scan2hostmem();
        // ...

        // launch sim. kernel(s)
        f->SimStep();

        // in meantime of simulation, store data
        fwrite(scan->data, sizeof(fas::data_t), scan->num_elements, dataf);

        // wait for simulation kernel(s) finish
        f->Finish();
    }

    std::cout << std::endl;
    fclose(dataf);

    std::cout << "Calculate RMS.\n";
    // all fields with RMS here
    f->FinishRms();
    // ...
    f->Finish();

    // store 3D array of RMS
    fas::data_t * rms = f->Map_rms_read();
    FILE* rmsf = fopen("rms0", "wb");
    // write header
    fwrite(&(f->size.x), sizeof(uint32_t), 1, rmsf);
    fwrite(&(f->size.y), sizeof(uint32_t), 1, rmsf);
    fwrite(&(f->size.z), sizeof(uint32_t), 1, rmsf);
    fwrite(rms, sizeof(fas::data_t), (size_t)f->size.x* f->size.y* f->size.z, rmsf);
    fclose(rmsf);

    delete scan;
    delete drv;
    delete f;
    delete dev;

    std::cout << std::endl;
    return 0;
}*/
