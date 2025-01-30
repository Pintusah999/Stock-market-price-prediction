// Password visibility toggle function
const showHiddenPass = (loginPass, loginEye) => {
  const input = document.getElementById(loginPass),
        iconEye = document.getElementById(loginEye);

  iconEye.addEventListener("click", () => {
    if (input.type === "password") {
      input.type = "text";
      iconEye.classList.add("ri-eye-line");
      iconEye.classList.remove("ri-eye-off-line");
    } else {
      input.type = "password";
      iconEye.classList.remove("ri-eye-line");
      iconEye.classList.add("ri-eye-off-line");
    }
  });
};
showHiddenPass("login-pass", "login-eye");

// Login form submission logic
document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const email = document.getElementById("login-email").value;
  const password = document.getElementById("login-pass").value;

  // Hardcoded credentials for demo purposes
  const validEmail = "user@example.com";
  const validPassword = "password123";

  // Check if the entered credentials are valid
  if (email === validEmail && password === validPassword) {
    // Redirect to another webpage after successful login
    window.location.href = "welcome.html"; // Replace with your desired webpage
  } else {
    alert("Invalid email or password. Please try again.");
  }
});
