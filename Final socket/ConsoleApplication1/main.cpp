#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0x40
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "SimConnect.h"


// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Global file stream for logging (optional)
std::ofstream logFile;

// Global socket for sending data to the Python client
SOCKET clientSocket = INVALID_SOCKET;

// Global SimConnect handle
HANDLE g_hSimConnect = nullptr;

// Define enums for data definitions and requests
enum DATA_DEFINE_ID {
    DEFINITION_FLIGHTDATA,
    DEFINITION_RESET    // For resetting aircraft state mid-flight
};
enum DATA_REQUEST_ID { REQUEST_FLIGHTDATA };

// Define client events for control inputs and reset (if needed)
enum CLIENT_EVENT_ID {
    EVENT_THROTTLE,
    EVENT_ELEVATOR,
    EVENT_AILERON,
    EVENT_RUDDER,
    EVENT_RESET  // Legacy reset event, if needed.
};

// Updated FlightData structure (7 values)
struct FlightData {
    double airspeed;       // "AIRSPEED INDICATED" (knots)
    double pitch;          // "PLANE PITCH DEGREES" (degrees)
    double bank;           // "PLANE BANK DEGREES" (degrees)
    double heading;        // "PLANE HEADING DEGREES TRUE" (degrees)
    double verticalSpeed;  // "VERTICAL SPEED" (feet per minute)
    double engineRPM;      // "GENERAL ENG RPM" (rpm)
    double altitude;       // "PLANE ALTITUDE" (feet)
};

// Structure for resetting the aircraft state (unchanged)
struct ResetData {
    double altitude;       // "PLANE ALTITUDE" in feet
    double bank;           // "PLANE BANK DEGREES" in degrees
    double heading;        // "PLANE HEADING DEGREES TRUE" in degrees
    double latitude;       // "PLANE LATITUDE" in degrees
    double longitude;      // "PLANE LONGITUDE" in degrees
    double pitch;          // "PLANE PITCH DEGREES" in degrees
    double airspeedTrue;   // "AIRSPEED TRUE" in knots
};

// Callback function to process simulator data.
void CALLBACK MyDispatchProc(SIMCONNECT_RECV* pData, DWORD cbData, void* pContext)
{
    switch (pData->dwID)
    {
    case SIMCONNECT_RECV_ID_SIMOBJECT_DATA:
    {
        auto* pObjData = (SIMCONNECT_RECV_SIMOBJECT_DATA*)pData;
        if (pObjData->dwRequestID == REQUEST_FLIGHTDATA)
        {
            FlightData* pFlight = reinterpret_cast<FlightData*>(&pObjData->dwData);
            DWORD timestamp = GetTickCount();
            char buffer[512];
            // Send timestamp then 7 values: airspeed, pitch, bank, heading, verticalSpeed, engineRPM, altitude.
            sprintf_s(buffer, "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                timestamp,
                pFlight->airspeed,
                pFlight->pitch,
                pFlight->bank,
                pFlight->heading,
                pFlight->verticalSpeed,
                pFlight->engineRPM,
                pFlight->altitude);
            std::string csvLine(buffer);
            send(clientSocket, csvLine.c_str(), (int)csvLine.length(), 0);

            static int counter = 0;
            if (++counter % 100 == 0)
            {
                std::cout << "[C++] Sent data (" << counter << "): " << csvLine;
            }
            if (logFile.is_open()) {
                logFile << csvLine;
                logFile.flush();
            }
        }
        break;
    }
    default:
        break;
    }
}

// ProcessControlCommand: if the command is "RESET", send ResetData.
void ProcessControlCommand(const std::string& command) {
    if (command == "RESET") {
        std::cout << "Reset command received. Updating aircraft state." << std::endl;
        ResetData resetData;
        resetData.altitude = 6000.0;
        resetData.bank = 0.0;
        resetData.heading = 270.0;
        resetData.pitch = 0.0;
        resetData.latitude = 56.046770;
        resetData.longitude = 12.768605;
        resetData.airspeedTrue = 150.0;  // Example value

        HRESULT hr = SimConnect_SetDataOnSimObject(
            g_hSimConnect,
            DEFINITION_RESET,
            SIMCONNECT_OBJECT_ID_USER,
            0,
            1,
            sizeof(resetData),
            &resetData
        );
        if (FAILED(hr)) {
            std::cerr << "SetData on RESET failed (0x" << std::hex << hr << ")." << std::endl;
        }
        return;
    }

    // Process other control commands.
    std::istringstream ss(command);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t colonPos = token.find(':');
        if (colonPos != std::string::npos) {
            std::string key = token.substr(0, colonPos);
            std::string valueStr = token.substr(colonPos + 1);
            double value = std::stod(valueStr);
            DWORD eventValue = static_cast<DWORD>(value * 16383);
            if (key == "THROTTLE") {
                SimConnect_TransmitClientEvent(g_hSimConnect, SIMCONNECT_OBJECT_ID_USER, EVENT_THROTTLE,
                    eventValue, SIMCONNECT_GROUP_PRIORITY_HIGHEST, SIMCONNECT_EVENT_FLAG_GROUPID_IS_PRIORITY);
            }
            else if (key == "ELEVATOR") {
                SimConnect_TransmitClientEvent(g_hSimConnect, SIMCONNECT_OBJECT_ID_USER, EVENT_ELEVATOR,
                    eventValue, SIMCONNECT_GROUP_PRIORITY_HIGHEST, SIMCONNECT_EVENT_FLAG_GROUPID_IS_PRIORITY);
            }
            else if (key == "AILERON") {
                SimConnect_TransmitClientEvent(g_hSimConnect, SIMCONNECT_OBJECT_ID_USER, EVENT_AILERON,
                    eventValue, SIMCONNECT_GROUP_PRIORITY_HIGHEST, SIMCONNECT_EVENT_FLAG_GROUPID_IS_PRIORITY);
            }
            else if (key == "RUDDER") {
                SimConnect_TransmitClientEvent(g_hSimConnect, SIMCONNECT_OBJECT_ID_USER, EVENT_RUDDER,
                    eventValue, SIMCONNECT_GROUP_PRIORITY_HIGHEST, SIMCONNECT_EVENT_FLAG_GROUPID_IS_PRIORITY);
            }
        }
    }
}

int main()
{
    // --- Winsock Initialization and Server Setup ---
    WSADATA wsData;
    if (WSAStartup(MAKEWORD(2, 2), &wsData) != 0) {
        std::cerr << "WSAStartup failed." << std::endl;
        return -1;
    }

    SOCKET listeningSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (listeningSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return -1;
    }

    sockaddr_in serverHint;
    serverHint.sin_family = AF_INET;
    serverHint.sin_port = htons(54000);
    serverHint.sin_addr.s_addr = INADDR_ANY;

    if (bind(listeningSocket, (sockaddr*)&serverHint, sizeof(serverHint)) == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << std::endl;
        closesocket(listeningSocket);
        WSACleanup();
        return -1;
    }

    if (listen(listeningSocket, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "Listen failed: " << WSAGetLastError() << std::endl;
        closesocket(listeningSocket);
        WSACleanup();
        return -1;
    }

    std::cout << "Waiting for a client to connect on port 54000..." << std::endl;
    clientSocket = accept(listeningSocket, (sockaddr*)NULL, (int*)NULL);
    if (clientSocket == INVALID_SOCKET) {
        std::cerr << "Accept failed: " << WSAGetLastError() << std::endl;
        closesocket(listeningSocket);
        WSACleanup();
        return -1;
    }
    std::cout << "Client connected!" << std::endl;
    closesocket(listeningSocket);

    // Set client socket to non-blocking mode.
    u_long mode = 1;
    if (ioctlsocket(clientSocket, FIONBIO, &mode) != 0) {
        std::cerr << "Failed to set non-blocking socket: " << WSAGetLastError() << std::endl;
        closesocket(clientSocket);
        WSACleanup();
        return -1;
    }

    // --- (Optional) Open CSV log file for internal logging ---
    logFile.open("FlightDataLog.csv", std::ios::out);
    if (logFile.is_open()) {
        // Header: Timestamp, Airspeed, Pitch, Bank, Heading, VerticalSpeed, EngineRPM, Altitude
        logFile << "Timestamp, Airspeed, Pitch, Bank, Heading, VerticalSpeed, EngineRPM, Altitude\n";
    }

    // --- Open a connection to SimConnect ---
    HRESULT hr = SimConnect_Open(&g_hSimConnect, "Flight Data Logging", NULL, 0, 0, 0);
    if (SUCCEEDED(hr))
    {
        // Map client events for control inputs.
        SimConnect_MapClientEventToSimEvent(g_hSimConnect, EVENT_THROTTLE, "THROTTLE_SET");
        SimConnect_MapClientEventToSimEvent(g_hSimConnect, EVENT_ELEVATOR, "ELEVATOR_SET");
        SimConnect_MapClientEventToSimEvent(g_hSimConnect, EVENT_AILERON, "AILERON_SET");
        SimConnect_MapClientEventToSimEvent(g_hSimConnect, EVENT_RUDDER, "RUDDER_SET");

        // Map legacy reset event (if needed)
        SimConnect_MapClientEventToSimEvent(g_hSimConnect, EVENT_RESET, "SimStart");

        // Register data definition for flight state logging.
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "AIRSPEED INDICATED", "knots", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "PLANE PITCH DEGREES", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "PLANE BANK DEGREES", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "PLANE HEADING DEGREES TRUE", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "VERTICAL SPEED", "feet per minute", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "GENERAL ENG RPM", "rpm", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_FLIGHTDATA, "PLANE ALTITUDE", "feet", SIMCONNECT_DATATYPE_FLOAT64);

        // Register data definition for resetting aircraft state.
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE ALTITUDE", "feet", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE BANK DEGREES", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE HEADING DEGREES TRUE", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE LATITUDE", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE LONGITUDE", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "PLANE PITCH DEGREES", "degrees", SIMCONNECT_DATATYPE_FLOAT64);
        SimConnect_AddToDataDefinition(g_hSimConnect, DEFINITION_RESET, "AIRSPEED TRUE", "knots", SIMCONNECT_DATATYPE_FLOAT64);

        // Request flight state data every simulation frame.
        SimConnect_RequestDataOnSimObject(
            g_hSimConnect,
            REQUEST_FLIGHTDATA,
            DEFINITION_FLIGHTDATA,
            SIMCONNECT_OBJECT_ID_USER,
            SIMCONNECT_PERIOD_SIM_FRAME
        );

        char recvBuffer[512];
        std::string receiveBuffer;

        bool quit = false;
        while (!quit)
        {
            SimConnect_CallDispatch(g_hSimConnect, MyDispatchProc, NULL);
            Sleep(1);

            ZeroMemory(recvBuffer, 512);
            int bytesReceived = recv(clientSocket, recvBuffer, 511, 0);
            if (bytesReceived > 0) {
                receiveBuffer.append(recvBuffer, bytesReceived);
                size_t newlinePos;
                while ((newlinePos = receiveBuffer.find('\n')) != std::string::npos) {
                    std::string command = receiveBuffer.substr(0, newlinePos);
                    receiveBuffer.erase(0, newlinePos + 1);
                    std::cout << "Received command: " << command << std::endl;
                    ProcessControlCommand(command);
                }
            }
            else if (bytesReceived == 0) {
                std::cout << "Client disconnected." << std::endl;
                closesocket(clientSocket);
                clientSocket = INVALID_SOCKET;
                break;
            }
            else {
                int error = WSAGetLastError();
                if (error != WSAEWOULDBLOCK) {
                    std::cerr << "recv failed: " << error << std::endl;
                    closesocket(clientSocket);
                    clientSocket = INVALID_SOCKET;
                    break;
                }
            }
        }
        SimConnect_Close(g_hSimConnect);
    }
    else {
        std::cerr << "Failed to open SimConnect connection!" << std::endl;
    }

    if (clientSocket != INVALID_SOCKET)
        closesocket(clientSocket);
    if (logFile.is_open())
        logFile.close();
    WSACleanup();


    return 0;
}
