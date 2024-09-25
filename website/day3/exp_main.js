// Declare parameters to be collected in the first HTML pages (consent form)
var participant_age = NaN
var participant_gender = NaN
var participant_turker = NaN

// Intialize JsPsych
const jsPsych = initJsPsych({
    // Initialize progress bar 
    show_progress_bar: false, //switched from true for jsPsych
    auto_update_progress_bar: false,

    // Save data on start to create a temporary file
    on_start: function() {
        logStart(params.task, params.id)
        saveData(params, jsPsych.data.get().csv(), {temporary :true})

        //save parameters
        saveData(params, params, {temporary : false, suffix: '_params'})
    },

    // Save the data in jsPsych is closed
    on_finish: function() {
        // Save the data at the end of the experiment -- EDIT KAI 4/27: EXCLUDED BC WE ARE ALREADY SAVING SEPARATELY
        saveData(params, jsPsych.data.get().csv(), {temporary : false})
        // Finish exp
        experiment_running = false
        // Show verification code at the end of the experiment
        writeHeader(task_name)
        goWebsite(html_vercode)
    },

    experiment_width : 1000,
})

// Function to launch experiment 
// (this is a wrapper to accomodate jsPsych with what has been done in the lab for the consent form, etc. )
function startExperiment() {

    // Initialize random seed with participant ID
    Math.seedrandom(participant_turker)

    // Define experiment parameters
    var params = returnExperimentParameters()
    console.log(params)

    //save parameters
    saveData(params, params, {temporary : false, suffix: '_params'})
    console.log('params saved')

    // Load timeline (sequence of blocs)
    const timeline = returnTimeline(params)
    params.participant_age = participant_age
    params.participant_gender = participant_gender
    params.participant_turker = participant_turker

    // Run timeline
    experiment_running = true
    jsPsych.run(
        timeline
    )
    
}

// Terminate experiment if the participant exit in fullscreen
function finishExperimentResize() {
    if (experiment_running) {
        saveData(params, jsPsych.data.get().csv(), {temporary : false, resize : true})
        writeHeader(task_name)
        goWebsite(html_errscreen)
    }
}