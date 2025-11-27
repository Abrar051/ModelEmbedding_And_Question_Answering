// Load Monaco Editor
require.config({ paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs" } });

require(["vs/editor/editor.main"], function () {

    // INITIALIZE EDITOR
    window.editor = monaco.editor.create(document.getElementById("editor-container"), {
        value: `# Write your code here`,
        language: "python",
        theme: "vs-dark",
        automaticLayout: true,
    });

});


// ----------------------------------------------
// SEND CODE TO BACKEND AND SHOW UNDERLINE ISSUES
// ----------------------------------------------
async function runScan() {

    const code = window.editor.getValue();

    let response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            code: code,
            language: "python"
        })
    });

    let result = await response.json();

    console.log("AI RESULT:", result);

    if (!result.issues) {
        alert("No issues found.");
        return;
    }

    applyDiagnostics(result.issues);
}


// ----------------------------------------------
// SHOW UNDERLINES LIKE VS CODE
// ----------------------------------------------
function applyDiagnostics(issues) {

    let markers = issues.map(issue => {

        let severity = monaco.MarkerSeverity.Info;

        if (issue.severity === "high") severity = monaco.MarkerSeverity.Error;
        if (issue.severity === "medium") severity = monaco.MarkerSeverity.Warning;
        if (issue.severity === "low") severity = monaco.MarkerSeverity.Info;

        return {
            severity: severity,
            message: issue.message + " | " + issue.suggestion,
            startLineNumber: issue.line,
            endLineNumber: issue.line,
            startColumn: issue.range ? issue.range[0] : 1,
            endColumn: issue.range ? issue.range[1] : 1000
        };
    });

    monaco.editor.setModelMarkers(window.editor.getModel(), "ai-diagnostics", markers);
}
