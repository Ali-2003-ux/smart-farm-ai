function startListen() {
    document.getElementById('status').innerHTML = "Listening...";
    // Simulation of Speech API for non-HTTPS localhost (often blocks mic)
    setTimeout(function () {
        document.getElementById('status').innerHTML = "Command Recognized: 'SCAN SECTOR 4'";
    }, 2000);
}
