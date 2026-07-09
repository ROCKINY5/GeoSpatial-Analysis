const { JSDOM } = require("jsdom");
const fs = require('fs');
const html = fs.readFileSync('geospatial_intelligence_module.html', 'utf8');

const dom = new JSDOM(html, { runScripts: "dangerously" });
const window = dom.window;

window.addEventListener('load', () => {
    try {
        console.log("Keys in siteData: ", Object.keys(window.siteData));
    } catch(e) {
        console.log(e);
    }
});
