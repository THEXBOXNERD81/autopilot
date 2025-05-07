#include <fstream>
#include <windows.h>
#include "SimConnect.h"

// Define an ID for your data definition:
enum DATA_DEFINE_ID {
    DEFINITION_AIRSPEED,
};

// Structure to receive the airspeed (adjust the type as needed)
struct AirspeedData {
    double airspeed;
};

// In your setup code, register the variable:
SimConnect_AddToDataDefinition(hSimConnect, DEFINITION_AIRSPEED, "AIRSPEED INDICATED", "knots", SIMCONNECT_DATATYPE_DOUBLE);


SimConnect_RequestDataOnSimObject(hSimConnect, REQUEST_AIRSPEED_DATA, DEFINITION_AIRSPEED, SIMCONNECT_OBJECT_ID_USER, SIMCONNECT_PERIOD_SECOND);


// Global file stream for logging
std::ofstream logFile;

// Dispatch callback for SimConnect events
void CALLBACK MyDispatchProc(SIMCONNECT_RECV* pData, DWORD cbData, void* pContext)
{
    switch (pData->dwID)
    {
    case SIMCONNECT_RECV_ID_SIMOBJECT_DATA:
    {
        auto* pObjData = (SIMCONNECT_RECV_SIMOBJECT_DATA*)pData;
        if (pObjData->dwRequestID == REQUEST_AIRSPEED_DATA)
        {
            // Cast received data to our structure
            AirspeedData* pAirspeed = (AirspeedData*)&pObjData->dwData;
            // Get a simple timestamp (e.g., from GetTickCount or another timer)
            DWORD timestamp = GetTickCount();
            // Write a CSV line: Timestamp, Airspeed
            if (logFile.is_open())
            {
                logFile << timestamp << ", " << pAirspeed->airspeed << "\n";
                logFile.flush(); // Optional: flush so data is saved immediately
            }
        }
        break;
    }
    default:
        break;
    }
}



int main()
{
    // Open the CSV file for logging (this file will be created in your working directory)
    logFile.open("SimDataLog.csv", std::ios::out);
    if (!logFile.is_open())
    {
        MessageBox(NULL, "Failed to open log file!", "Error", MB_OK);
        return -1;
    }
    // Write a header line
    logFile << "Timestamp, Airspeed\n";

    // Open a connection to SimConnect
    HANDLE hSimConnect = nullptr;
    HRESULT hr = SimConnect_Open(&hSimConnect, "Airspeed Logging", NULL, 0, 0, 0);
    if (SUCCEEDED(hr))
    {
        // Add the data definition for airspeed (as shown earlier)
        SimConnect_AddToDataDefinition(hSimConnect, DEFINITION_AIRSPEED, "AIRSPEED INDICATED", "knots", SIMCONNECT_DATATYPE_DOUBLE);
        // Request data on the user’s aircraft, updating every second
        SimConnect_RequestDataOnSimObject(hSimConnect, REQUEST_AIRSPEED_DATA, DEFINITION_AIRSPEED, SIMCONNECT_OBJECT_ID_USER, SIMCONNECT_PERIOD_SECOND);

        // Main message loop: process events until you decide to quit
        bool quit = false;
        while (!quit)
        {
            SimConnect_CallDispatch(hSimConnect, MyDispatchProc, NULL);
            Sleep(100); // Sleep briefly to avoid high CPU usage
            // (Implement your own exit condition as needed)
        }
        SimConnect_Close(hSimConnect);
    }
    else
    {
        MessageBox(NULL, "Failed to open SimConnect connection!", "Error", MB_OK);
    }

    // Close the log file
    if (logFile.is_open())
        logFile.close();

    return 0;
}
