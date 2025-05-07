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
#include <vector>    
#include <thread>// ← new
#include "SimConnect.h"

// Link with Ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

// Global file stream for logging (optional)
std::ofstream logFile;

// Replace single socket with vector of client sockets
std::vector<SOCKET> clientSockets;

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

// ProcessControlCommand: if the command is "RESET", send ResetData.
void ProcessControlCommand(const std::string& command) {
    if (command == "RESET") {
        std::cout << "Reset command received. Updating aircraft state." << std::endl;
        ResetData resetData;
        resetData.altitude = 3000.0;
        resetData.bank = 0.0;
        resetData.heading = 270.0;
        resetData.pitch = 0.0;
        resetData.latitude = 56.046770;
        resetData.longitude = 12.768605;
        resetData.airspeedTrue = 100.0;  // Example value

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

// Broadcast a line to all connected clients
void broadcast_line(const std::string& line) {
    for (auto& sock : clientSockets) {
        if (sock != INVALID_SOCKET) {
            send(sock, line.c_str(), (int)line.size(), 0);
        }
    }
}

// In your dispatch proc, replace send(clientSocket,…) with broadcast_line(...)
void CALLBACK MyDispatchProc(SIMCONNECT_RECV* pData, DWORD cbData, void* pContext) {
    if (pData->dwID == SIMCONNECT_RECV_ID_SIMOBJECT_DATA) {
        auto* pObjData = (SIMCONNECT_RECV_SIMOBJECT_DATA*)pData;
        if (pObjData->dwRequestID == REQUEST_FLIGHTDATA) {
            FlightData* pFlight = reinterpret_cast<FlightData*>(&pObjData->dwData);
            DWORD timestamp = GetTickCount();
            char buffer[512];
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

            // send to all clients
            broadcast_line(csvLine);

            static int counter = 0;
            if (++counter % 100 == 0) {
                std::cout << "[C++] Sent data (" << counter << "): " << csvLine;
            }
            if (logFile.is_open()) {
                logFile << csvLine;
                logFile.flush();
            }
        }
    }
}

// A helper to process incoming commands on a single socket
void handle_client_commands(SOCKET sock) {
    char recvBuffer[512];
    std::string buffer;
    while (true) {
        ZeroMemory(recvBuffer, sizeof(recvBuffer));
        int bytes = recv(sock, recvBuffer, sizeof(recvBuffer) - 1, 0);
        if (bytes > 0) {
            buffer.append(recvBuffer, bytes);
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                std::string cmd = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);
                std::cout << "Received command: " << cmd << std::endl;
                ProcessControlCommand(cmd);
            }
        }
        else if (bytes == 0) {
            // client cleanly disconnected
            std::cout << "Client disconnected\n";
            break;
        }
        else {
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK) {
                // no data right now
                Sleep(50);
                continue;
            }
            else {
                std::cerr << "recv error: " << err << "\n";
                break;
            }
        }
    }
    closesocket(sock);
}

int main()
{
    // 1) Winsock init
    WSADATA wsData;
    if (WSAStartup(MAKEWORD(2, 2), &wsData) != 0) {
        std::cerr << "WSAStartup failed.\n";
        return -1;
    }

    // 2) Listening socket
    SOCKET listenSock = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed: " << WSAGetLastError() << "\n";
        WSACleanup();
        return -1;
    }

    sockaddr_in hint = {};
    hint.sin_family = AF_INET;
    hint.sin_port = htons(54000);
    hint.sin_addr.s_addr = INADDR_ANY;

    if (bind(listenSock, (sockaddr*)&hint, sizeof(hint)) == SOCKET_ERROR) {
        std::cerr << "Bind failed: " << WSAGetLastError() << "\n";
        closesocket(listenSock);
        WSACleanup();
        return -1;
    }
    if (listen(listenSock, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "Listen failed: " << WSAGetLastError() << "\n";
        closesocket(listenSock);
        WSACleanup();
        return -1;
    }

    std::cout << "Waiting for 2 clients on port 54000...\n";
    std::vector<SOCKET> clientSockets;
    while (clientSockets.size() < 2) {
        SOCKET s = accept(listenSock, nullptr, nullptr);
        if (s != INVALID_SOCKET) {
            // make recv() nonblock
            u_long mode = 1;
            ioctlsocket(s, FIONBIO, &mode);
            clientSockets.push_back(s);
            std::cout << "Client #" << clientSockets.size() << " connected.\n";
        }
        else {
            Sleep(1);
        }
    }
    closesocket(listenSock);

    // 3) Optional log file
    logFile.open("FlightDataLog.csv");
    if (logFile.is_open())
        logFile << "Timestamp,Airspeed,Pitch,Bank,Heading,VerticalSpeed,EngineRPM,Altitude\n";

    // 4) SimConnect open
    if (FAILED(SimConnect_Open(&g_hSimConnect, "Flight Data Logging", NULL, 0, 0, 0))) {
        std::cerr << "SimConnect_Open failed\n";
        for (auto& sock : clientSockets) closesocket(sock);
        WSACleanup();
        return -1;
    }
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

        // 6) Spin off one thread per client to handle incoming commands
        for (auto sock : clientSockets) {
            std::thread(handle_client_commands, sock).detach();
        }

        // 7) Main dispatch loop: SimConnect → MyDispatchProc → broadcast_line()
        while (true) {
            SimConnect_CallDispatch(g_hSimConnect, MyDispatchProc, nullptr);
            Sleep(1);
        }

        // never reached
        return 0;
    }
}