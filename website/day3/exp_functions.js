// Kai Sandbrink
// 2023.04.05
// This script contains functions for the OBE game

// GLOBAL STYLE FUNCTIONS
function set_html_style_instructions() {
    document.body.style.color = 'white'; // font color
    document.body.style.backgroundImage = "url('stim/background/casino.png')";
    document.body.style.backgroundRepeat = "no-repeat";
    document.body.style.backgroundPosition = "center"
    document.body.style.width="750px";
    document.body.style.margin="auto";
    document.body.style.backgroundColor = 'black'; // background color
}

function set_html_style_normal() {
    document.body.style.backgroundColor = 'white'; // background color
    document.body.style.backgroundImage = "none";
    document.body.style.width="100%";
    document.body.style.margin="auto";
    document.body.style.color = 'black'; // font color
}

function getEfficacyColor(value, alpha) {

    // Make sure the input is within the bounds [0, 1]
    if (value < 0 || value > 1 || alpha < 0 || alpha > 1) {
        throw new Error('Input values must be floats between 0 and 1.');
    }

    // Define the RGB values for the start and end colors
    const startColor = { r: 255, g: 0, b: 0 };  // Red
    const endColor = { r: 255, g: 255, b: 0 };  // Yellow

    // Calculate the interpolated color
    const r = Math.round(startColor.r + value * (endColor.r - startColor.r));
    const g = Math.round(startColor.g + value * (endColor.g - startColor.g));
    const b = Math.round(startColor.b + value * (endColor.b - startColor.b));

    // Convert the color to an RGBA string
    const color = `rgba(${r}, ${g}, ${b}, ${alpha})`;

    return color;
}


function reset_style() {
    document.body.style.backgroundColor = 'white'; // background color
    document.body.style.color = 'black'; 
}

// SMALLER STYLE ELEMENTS
function format_image_button(file_stem) {
    return `<img style="max-height:200px;" src="stim/` + file_stem + `.JPG">`;
}

function chooseColor(arm){
    if (arm == 0) {
        return 'Green';
    }
    else {
        return 'Purple';
    }
}

function format_quiz_feedback(questions, answers, solutions) {

    feedback = `<p>Solutions</p>`

    for (let i = 0; i < answers.length; i++) {
        if(answers[i]==solutions[i]) {
            feedback += `<p style='text-align:left'><span style="color:green"><br>Q` + (i + 1) + ` Correct! `;
        }
        else
        {
            feedback += `<p style='text-align:left'><span style="color:red"><br>Q` + (i + 1) + ` Incorrect! `;
        }

        feedback += `</span>` + questions[i] + `<br>Solution: ` + solutions[i] + `</p>`;

    }

    return feedback;
}

// SCREEN AND ELEMENT CREATION

function create_quiz(quiz_prompts, quiz_options, quiz_idxs_correct, solutions_pictures) {
    /* Create question and feedback screens that together constitute a quiz

    Arguments
    ---------
    quiz_prompts : array or strings [n_prompts,]
    quiz_options : array of arrays of strings [n_prompts, n_options_per_prompt]
    quiz_idxs_correct : array of integers [n_prompts,]
    solutions_pictures : array of strings [n_prompts,]

    Returns
    -------
    quiz_timeline : array of jsPsych Timeline objects [2]

    */

    var n_questions = quiz_prompts.length

    var quiz_timeline = []

    //console.log(quiz_options)

    var quiz_questions = []
    for(let i = 0; i < n_questions; i++) {
        quiz_questions.push( 
            {
                prompt: `Q` + (i+1).toString() + `: ` + quiz_prompts[i],
                options: quiz_options[i],
                required: true,
            }
        )
    }

    var give_quiz = {
        type: jsPsychSurveyMultiChoice,
        questions: quiz_questions,
        randomize_question_order: false
    };
    quiz_timeline.push(give_quiz)

    var quiz_solutions = [];
    for(let i = 0; i < n_questions; i++) {
        sol = quiz_options[i][quiz_idxs_correct[i]];
        quiz_solutions.push(sol);
    }

    var quiz_feedback = {
        type: jsPsychHtmlButtonResponse,
        choices: ["Continue"],
        stimulus: function() {
            q0 = jsPsych.data.get().last().values()[0].response.Q0;
            q1 = jsPsych.data.get().last().values()[0].response.Q1;
            q2 = jsPsych.data.get().last().values()[0].response.Q2;

            if(n_questions > 3) {
                q3 = jsPsych.data.get().last().values()[0].response.Q3;
                if (n_questions > 4) {
                    q4 = jsPsych.data.get().last().values()[0].response.Q4;
                    answers = [q0, q1, q2, q3, q4];	
                }
                else{
                    answers = [q0, q1, q2, q3];	
                }
            }
            else {
                answers = [q0, q1, q2];
            }

            return format_quiz_feedback(quiz_prompts, answers, quiz_solutions);
        },
        margin_vertical: '18px',
    };
    quiz_timeline.push(quiz_feedback)

    return quiz_timeline;

}