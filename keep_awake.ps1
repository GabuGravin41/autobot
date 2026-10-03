$code = @'
using System;
using System.Runtime.InteropServices;
public static class SleepUtil {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@
Add-Type -TypeDefinition $code -Language CSharp
$flags = [Convert]::ToUInt32("80000003", 16)
$res = [SleepUtil]::SetThreadExecutionState($flags)
Write-Host "SetThreadExecutionState active with flags 0x80000003: $res"

# Prevent monitor and standby timeout on AC power
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
Write-Host "Powercfg settings updated to never timeout."

$counter = 0
while ($true) {
    [SleepUtil]::SetThreadExecutionState($flags) | Out-Null
    Start-Sleep -Seconds 30
    $counter += 30
    
    # Every 30 minutes (1800 seconds), refresh the Kaggle access token
    if ($counter -ge 1800) {
        $counter = 0
        try {
            $t = kaggle auth print-access-token 2>$null
            if ($t -and $t.Length -gt 20) {
                $t.Trim() | Out-File -FilePath "$HOME\.kaggle\access_token" -Encoding ascii -NoNewline
                Write-Host "Kaggle access token refreshed automatically at $(Get-Date -Format 'HH:mm:ss')."
            }
        } catch {
            Write-Host "Token refresh attempt failed: $_"
        }
    }
}
