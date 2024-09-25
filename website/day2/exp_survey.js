function createSurvey(params) { 
    var survey_timeline = [];

    var welcome = {
        type: jsPsychInstructions,
        pages : [ `<h1>Post-Questionnaires</h1><p>We ask that you fill out a few short accompanying questionnaires.</p>`,
        ],

        show_clickable_nav : true
    }

    survey_timeline.push(welcome)

    counter = 1;

    for (let survey in transdiagnostic_qs) {
        survey_page = {
            type: jsPsychSurveyLikert,
            preamble: `<h2>Questionnaire ` + counter + `/` + Object.keys(transdiagnostic_qs).length + `</h2><p>` + transdiagnostic_qs[survey].preamble + `</p>`,
            questions: transdiagnostic_qs[survey].questions.map(question => {
            return {
                prompt: question.prompt,
                labels: question.labels,
                name: question.question_id,
                //reverse: question.reverse_coded,
                required : true,
            };}),
            on_finish : function (data) { 
                data.id = params.id
            }
        }
        
        counter ++;

        survey_timeline.push(survey_page);
    }
    
    var bloc = {
        timeline : survey_timeline,
        on_timeline_finish: function(data) { saveData(params, jsPsych.data.getLastTimelineData().csv(), {temporary : false, suffix : '_survey'})}
    }

    return bloc;

}