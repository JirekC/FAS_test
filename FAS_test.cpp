#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "FAS.cl/fas.hpp"
#include "nlohmann/json.hpp" // JSON parser for project files
#include <stdlib.h> // system() call
#include <unistd.h> // readlink()

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

std::string getexepath()
{
    char result[ 1024 ];
    ssize_t count = readlink( "/proc/self/exe", result, sizeof(result) );
    std::string path( result, (count > 0) ? count : 0 );
    size_t found = path.find_last_of("/\\");
    path = path.substr(0,found);
    return path;
}

int main(int argc, char* argv[])
{
    std::string exec_path = getexepath(); // path to executable of this process
    fas::device* dev; // TODO: allow use of multiple devices - each field can select GPU ...
    int sim_steps = 10e3; // total # of simulation steps
    fas::data_t dt; // time-step [sec]
    fas::data_t dx; // space-step [m]
    std::vector<fas::material> materials; // all used materials - common for whole project
    std::vector<fas::field*> fields; // simulation fields - TODO: connect together by boundaries ...
    std::vector<fas::driver*> drivers; // drivers from all fields together
    std::vector<fas::scanner*> scanners; // scanners from all fields together

    if (argc < 2)
    {
        std::cerr << "ERR: no input file.\n";
        std::cerr << "Using: FAS [project_file.json]\n";
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
//    std::cin >> plat_idx;
plat_idx = 0;
    if (plat_idx < 0 || plat_idx >= platforms.size()) {
        std::cerr << "ERR: not valid platform index\n";
        return -1;
    }
    platforms[plat_idx].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    std::cout << "Select device from platform (index): ";
//    std::cin >> dev_idx;
dev_idx = 0;
    if (dev_idx < 0 || dev_idx >= devices.size()) {
        std::cerr << "ERR: not valid device index\n";
        return -1;
    }

    /********************************************************/
    /*** prepare device : compile program, create kernels ***/
    /********************************************************/
    try {
        dev = new fas::device(devices[0], "../lib/FAS.cl/fas.cl");
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
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

        /*** overal project parameters ***/
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

        /*** load & recalculate all materials used in project ***/
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

        /*** load fields ***/
        for(auto jf : jproject["fields"])
        {
            std::cout << "Creating new field: ";
            if((val = jf["name"]).is_string())
                std::cout << std::string(val);
            std::cout << "\n";
            fas::vec3<uint32_t> size;
            try
            {
                size = parse_vec3<double>(jf["size"]) * (1.0 / dx); // load size in [m] and recalc to simulation units
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: size is not specified:\n" + e.what());
            }

            /*** init field ***/
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

            /*** create drivers ***/
            cntr = 1;
            {
                std::string vox_file_name = "F" + std::to_string(fields.size()) + "_drv.ui8";
                std::string cmd_line = "../STL2VOX/STL2VOX";
                cmd_line += " -o" + vox_file_name;
                cmd_line += " -sx" + std::to_string(f->size.x); // in [elements]
                cmd_line += " -sy" + std::to_string(f->size.y);
                cmd_line += " -sz" + std::to_string(f->size.z);
                cmd_line += " -dx" + std::to_string(dx); // size of single element
                for( auto jdriver : jf["drivers"] )
                {
                    // parse path
                    if(!(val = jdriver["model"]).is_string())
                    {
                        throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: " \
                            "driver [" + std::to_string(cntr - 1) + "]: \"model\" not specified\n");
                    }
                    cmd_line += " " + std::string(val);
                    cmd_line += " " + std::to_string(cntr); // material identifier used to identify individual drivers

                    cntr++;
                }
                if(system(cmd_line.c_str()) != 0)
                {
                    throw std::runtime_error("STL2VOX failed, check all driver's path\n");
                }
                fas::object::LoadVoxelMap(*f, vox_file_name.c_str()); // load voxel map to GPU
                for(int i = 1; i < cntr; i++) {
                    fas::driver *drv = new fas::driver(*f);
                    drv->CollectElements(i);
                    drivers.push_back(drv);
                }
                f->Clear();
                f->cl_queue.finish(); // wait for all work done (on device side)
            }

            /*** create scanners ***/
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
                    rotation = parse_vec3<double>(jscan["rotation"]);
                    // load and recalc position and size from meters to simulation units
                    position = parse_vec3<double>(jscan["position"]) * (1.0 / dx);
                    size = parse_vec2<double>(jscan["size"]) * (1.0 / dx);
                    // std::cout<< "Scanner position: " << std::to_string(position.x) << ", " << std::to_string(position.y) << ", " << std::to_string(position.z) << "\n";
                    // std::cout<< "Scanner size: " << std::to_string(size.x) << ", " << std::to_string(size.y) << "\n";
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

            /*** create .stl models inside field ***/
            cntr = 0;
            {
                std::string path;
                uint8_t mat_id;
                std::string vox_file_name = "F" + std::to_string(fields.size()) + ".ui8";
                std::string cmd_line = "../STL2VOX/STL2VOX";
                cmd_line += " -o" + vox_file_name;
                cmd_line += " -sx" + std::to_string(f->size.x); // in [elements]
                cmd_line += " -sy" + std::to_string(f->size.y);
                cmd_line += " -sz" + std::to_string(f->size.z);
                cmd_line += " -dx" + std::to_string(dx); // size of single element
                for( auto jmodel : jf["models"] )
                {
                    // parse path
                    if(!(val = jmodel["path"]).is_string())
                    {
                        throw std::runtime_error("Field [" + std::to_string(fields.size()) + "]: " \
                            "model [" + std::to_string(cntr) + "]: \"path\" not specified\n");
                    }
                    path = val;
                    // parse material id
                    if(!(val = jmodel["material_id"]).is_number_unsigned())
                    {
                        std::cerr << "Field [" + std::to_string(fields.size()) + "]: " \
                            "model [" + std::to_string(cntr) + "]: \"material_id\" not specified - assuming #0\n";
                        mat_id = 0;
                    }
                    else
                    {
                        mat_id = val;
                        if(mat_id > 255)
                        {
                            std::cerr  << "Field [" + std::to_string(fields.size()) + "]: " \
                                "model [" + std::to_string(cntr) + "]: \"material_id\" greater than 255 - #0 will be used instead\n";
                            mat_id = 0;
                        }
                    }
                    cmd_line += " " + path;
                    cmd_line += " " + std::to_string(mat_id);
                    cntr++;
                }
                if(system(cmd_line.c_str()) != 0)
                {
                    throw std::runtime_error("STL2VOX failed, check all model's path\n");
                }
                fas::object::LoadVoxelMap(*f, vox_file_name.c_str()); // load voxel map to GPU
                f->cl_queue.finish(); // wait for all work done (on device side)
            }

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

            // all drivers here
            for(auto d : drivers)
            {
                d->Drive(signal);
            }
            for(auto f : fields)
            {
                f->cl_queue.enqueueBarrierWithWaitList();
            }

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

    delete dev;

    // call destructors
    // delete[] scanners.data();
    // delete[] fields.data();

    std::cout << "Finished.\n";
    return 0;
}
