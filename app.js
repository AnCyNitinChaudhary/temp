require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const fs = require('fs');
const chokidar = require('chokidar');
const fsm = require('fs/promises');
const BASE_URL = process.env.BASE_URL;
const port = process.env.PORT;
const http = require('http');
const socketIo = require('socket.io');
app.use(express.json());


app.get('/home', (req, res)=>{
    res.sendFile(__dirname + '/homePage.html');
})


const jsonFilePath = 'locationData.json';

const readJsonFile = async () => {
    try {
        const data = await fsm.readFile(jsonFilePath, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('Error reading JSON file:', error);
        return null;
    }
};

const watcher = chokidar.watch(jsonFilePath);
watcher.on('change', async () => {
    console.log('JSON file changed. Reloading data...');
    const jsonData = await readJsonFile();
    io.emit('data-update', jsonData); // Emit the updated data to connected clients
});

let temp = 0;
app.get('/api/realtime-data', async (req, res) => {
    const jsonData = await readJsonFile();
    res.json(jsonData);
    // res.json(temp);
    // console.log("hey", temp);
});
let receivedRandomNumber = 0;
app.use(bodyParser.json());
app.post('/api/random', (req, res) => {
    receivedRandomNumber = req.body.randomNumber;
    console.log('Received random number from frontend:', receivedRandomNumber);
    res.json({ message: 'Random number received successfully' });
});

app.get('/home/teacher', (req, res) => {
    function generateRandomNumber() {
        const min = 100;
        const max = 999;
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
temp = generateRandomNumber();
    res.sendFile(__dirname + '/mam.html');
    console.log(generateRandomNumber());
});
// /indentify
// //js ka code
const server = app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});

// Add Socket.IO setup
const io = require('socket.io')(server);

// Listen for connection events
io.on('connection', (socket) => {
    console.log('Client connected');
});

app.get('/api/coordinates', (req, res) => {
    res.sendFile(__dirname + '/index.html');    
});

app.get('/admin', (req, res) => {
    res.sendFile(__dirname + '/adminPage.html')
});

app.get('/home/student', (req, res) => {
    res.sendFile(__dirname + '/i.html');
    // console.log(res.json());
});
// app.use(bodyParser.json());

// const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


app.use(express.static('public')); // Serve static files from the 'public' directory


// API endpoint to receive data from the student page
app.post('/home/teacher', (req, res) => {
    const data = req.body;

    // Broadcast the received data to all connected clients (teacher page)
    io.emit('dataUpdate', data);

    res.sendStatus(200);
});


app.get('/index', (req, res)=>{
    res.sendFile(__dirname + '/tt.html');
})

app.post('/home/student', (req, res) => {
    const enteredCode = req.body.code;

// console.log()
    // Check if the entered code is valid
    // if (validCodes.has(enteredCode)) {
    //     res.redirect(`/home/${enteredCode}`);
    // } else {
    //     res.send('Invalid code. Please try again.');
    // }

    console.log("entered code is: ",enteredCode);
    console.log("i got: ", receivedRandomNumber);
    if(enteredCode == receivedRandomNumber)  res.redirect(`/api/coordinates`);
    // else {res.send('code not matched');}
        else {res.send('code not matched');}
});




let locationData = [];



// this is the final code till now

app.post('/api/sendLocation', (req, res) => {
    // console.log("andar");
    const receivedLocation = req.body;

    const enrollmentNumber = receivedLocation.enrollment;
    // console.log("enrollment numberrr: ", enrollmentNumber);
    
    if (isValidEnrollmentNumber(enrollmentNumber)) {
        const existingEntryIndex = locationData.findIndex(entry => entry.enrollment === enrollmentNumber);

        if (existingEntryIndex === -1) {
            locationData.push(receivedLocation);
            saveDataToFile();
            notifyAdmin(enrollmentNumber);

            res.status(200).json({ attendanceMarked: true, message: 'Location data saved successfully' });
        } else {
            res.status(400).json({ attendanceMarked: false, message: 'Location data already exists for this enrollment number' });
        }
    } else {
        res.status(400).json({ attendanceMarked: false, message: 'Invalid enrollment number' });
    }
});



// app.post('/api/sendLocation', (req, res) => {
   
//     if (!apiActivated) {
//         return res.status(400).json({ attendanceMarked: false, message: 'API is deactivated' });
//     }

//     const receivedLocation = req.body;
//     const enrollmentNumber = receivedLocation.enrollment;

//     if (isValidEnrollmentNumber(enrollmentNumber)) {
//         const existingEntryIndex = locationData.findIndex(entry => entry.enrollment === enrollmentNumber);

//         if (existingEntryIndex === -1) {
//             locationData.push(receivedLocation);
//             saveDataToFile();
//             notifyAdmin(enrollmentNumber);

//             return res.status(200).json({ attendanceMarked: true, message: 'Location data saved successfully' });
//         } else {
//             return res.status(400).json({ attendanceMarked: false, message: 'Location data already exists for this enrollment number' });
//         }
//     } else {
//         return res.status(400).json({ attendanceMarked: false, message: 'Invalid enrollment number' });
//     }
// });

function isValidEnrollmentNumber(enrollmentNumber) {
    return /^\d{10}$/.test(enrollmentNumber);
}



function saveDataToFile() {
    fs.writeFileSync('locationData.json', JSON.stringify(locationData, null, 2), 'utf-8');
}

function notifyAdmin(enrollmentNumber) {
    console.log(`Attendance marked for enrollment number: ${enrollmentNumber}`);
}

// app.listen(port, () => {
//     console.log(`Server is running at http://localhost:${port}`);
// });



