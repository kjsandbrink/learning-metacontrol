// Define main experimental timeline
// This adds all the trials to a timeline array
function returnTimeline(params) {
    let timeline = []
    // Starting bloc 
    timeline.push(blocStart(params))
    // One bloc
    timeline.push(blocMain(params))
    // Survey
    timeline.push(createSurvey(params))
    // End bloc
    timeline.push(blocEnd(params))
    return timeline
}
