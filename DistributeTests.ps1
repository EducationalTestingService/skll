<#
.SYNOPSIS
    Distribute the tests in VSTS pipeline across multiple agents
.DESCRIPTION
    This script divides test files across multiple agents for running on Azure DevOps.
    It is adapted from the script in this repository:
    https://github.com/PBoraMSFT/ParallelTestingSample-Python/blob/master/DistributeTests.ps1

    The distribution is basically identical to the way we do it in .gitlab-ci.yaml
#>

$tests = Get-ChildItem .\tests\ -Filter "test*.py" # search for test files with specific pattern.
$totalAgents = [int]$Env:SYSTEM_TOTALJOBSINPHASE # standard VSTS variables available using parallel execution; total number of parallel jobs running
$agentNumber = [int]$Env:SYSTEM_JOBPOSITIONINPHASE  # current job position
$testCount = $tests.Count

# below conditions are used if parallel pipeline is not used. i.e. pipeline is running with single agent (no parallel configuration)
if ($totalAgents -eq 0) {
    $totalAgents = 1
}
if (!$agentNumber -or $agentNumber -eq 0) {
    $agentNumber = 1
}

Write-Host "Total agents: $totalAgents"
Write-Host "Agent number: $agentNumber"
Write-Host "Total tests: $testCount"

$testsToRun= @()

if ($agentNumber -eq 1) {
    $testsToRun = $testsToRun + "test_featureset"
    $testsToRun = $testsToRun + "test_commandline_utils"
    $testsToRun = $testsToRun + "test_custom_metrics"
    $testsToRun = $testsToRun + "test_voting_learners_api_5"
}
elseif ($agentNumber -eq 2) {
    $testsToRun = $testsToRun + "test_output"
    $testsToRun = $testsToRun + "test_voting_learners_api_4"
}
elseif ($agentNumber -eq 3) {
    $testsToRun = $testsToRun + "test_regression"
    $testsToRun = $testsToRun + "test_voting_learners_api_2"
}
elseif ($agentNumber -eq 4) {
    $testsToRun = $testsToRun + "test_input"
    $testsToRun = $testsToRun + "test_preprocessing"
    $testsToRun = $testsToRun + "test_metrics"
    $testsToRun = $testsToRun + "test_custom_learner"
    $testsToRun = $testsToRun + "test_logging_utils"
    $testsToRun = $testsToRun + "test_examples"
    $testsToRun = $testsToRun + "test_voting_learners_api_1"
    $testsToRun = $testsToRun + "test_voting_learners_expts_1"
}
elseif ($agentNumber -eq 5) {
    $testsToRun = $testsToRun + "test_classification"
    $testsToRun = $testsToRun + "test_cv"
    $testsToRun = $testsToRun + "test_ablation"
    $testsToRun = $testsToRun + "test_voting_learners_expts_4"
}
elseif ($agentNumber -eq 6) {
    $testsToRun = $testsToRun + "test_voting_learners_api_3"
    $testsToRun = $testsToRun + "test_voting_learners_expts_2"
    $testsToRun = $testsToRun + "test_voting_learners_expts_3"
    $testsToRun = $testsToRun + "test_voting_learners_expts_5"
}

# join all test files seperated by space. pytest runs multiple test files in following format pytest test1.py test2.py test3.py
$testFiles = $testsToRun -Join " "
Write-Host "Test files $testFiles"
# write these files into variable so that we can run them using pytest in subsequent task.
Write-Host "##vso[task.setvariable variable=pytestfiles;]$testFiles"
