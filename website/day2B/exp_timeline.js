// Define main experimental timeline
// This adds all the trials to a timeline array
function returnTimeline(params) {
    let timeline = []
    timeline.push(blocStart(params))
    timeline.push(blocMain(params))
    // End bloc
    timeline.push(blocEnd(params))
    return timeline
}
