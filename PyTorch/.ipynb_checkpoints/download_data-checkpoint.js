var unirest = require("unirest");
const fs = require('fs');

var req = unirest("GET", "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data");

req.query({
    "symbol": "WFC",
    "region": "US"
});

req.headers({
    "x-rapidapi-key": "da8669d8efmsh82222b4712b7dbdp1189a8jsn9845ce331e2f",
    "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
    "useQueryString": true
});


req.end(function (res) {
    if (res.error) throw new Error(res.error);
    writeF(JSON.stringify(res.body))
});

function writeF(data) {
    var data = new Uint8Array(Buffer.from(data));
    fs.writeFile('WFCHistorical.txt', data, (err) => {
      if (err) throw err;
      console.log('The file has been saved!');
    });
}