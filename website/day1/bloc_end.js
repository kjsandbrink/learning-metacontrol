function blocEnd(params) {
    let bloc = {
        timeline: [
            
            // END WITHOUT SURVEY
            {
                type: jsPsychHtmlButtonResponse,
                choices: ['End experiment'],
                stimulus: function() {

                    // Compute percentage correct responses
                    let rew = jsPsych.data.get().filter({end_ep: true}).last(1).values()[0].total_rewards_tally
        
                    return `<p>Thanks for completing the experiment! Your total reward is ${rew}</b>. Congrats! <br>  
                    Click to complete the experiment and access your verification code. </p>`
                },
                on_finish: function(data) {
                    // Save the data at the end of the experiment
                    saveData(params, jsPsych.data.get().csv(), temporary = false)
                    // Finish exp
                    experiment_running = false
                },
            },
            

            {
                type: jsPsychFullscreen,
                fullscreen_mode: false
            }


        ]
    }
    return bloc
}
