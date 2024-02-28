const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;

// Valid class codes
// const validCodes = new Set(); // Add valid class codes here

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public')); // Serve static files from the 'public' directory

// Home page route
app.get('/home', (req, res) => {
    res.sendFile(__dirname + '/i.html');
});

// Handle form submission
app.post('/home', (req, res) => {
    const enteredCode = req.body.code;

    // Check if the entered code is valid
    // if (validCodes.has(enteredCode)) {
    //     res.redirect(`/home/${enteredCode}`);
    // } else {
    //     res.send('Invalid code. Please try again.');
    // }
    console.log("entered code is: ",enteredCode);
    if(enteredCode == 234) res.redirect(`/home/${enteredCode}`);
    else {res.send('code not matched');}
});

// Dynamic route for valid codes
// app.get('/home/:code', (req, res) => {
    
//     const code = req.params.code;
//     res.send(`Welcome to the home page for class Large Scale Database!`);
// });















app.listen(port, () => {
    console.log(`Student Page Server is running at http://localhost:${port}`);
});
