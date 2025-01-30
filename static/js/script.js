let navbar = document.querySelector('.header .navbar');
let menuBtn = document.querySelector('#menu');

menuBtn.onclick = () =>{
   menuBtn.classList.toggle('fa-times');
   navbar.classList.toggle('active');
};

window.onscroll = () =>{
   menuBtn.classList.remove('fa-times');
   navbar.classList.remove('active');
};

let themeBtn = document.querySelector('#theme-btn');

themeBtn.onclick = () =>{
    themeBtn.classList.toggle('fa-sun');

    if(themeBtn.classList.contains('fa-sun')){
        document.body.classList.add('active');
    }else{
        document.body.classList.remove('active');
    }

};

var swiper = new Swiper(".gallery-slider", {
    effect: "coverflow",
    grabCursor: true,
    centeredSlides: true,
    slidesPerView: "auto",
    coverflowEffect: {
      rotate: 20,
      stretch: 0,
      depth: 100,
      modifier: 1,
      slideShadows: true,
    },
    loop:true,
    autoplay:{
      delay: 3000,
      disableOnInteraction:false,
    }
  });

  var swiper = new Swiper(".review-slider", {
    loop:true,
    grabCursor: true,
    spaceBetween: 20,
    centeredSlides: true,
    autoplay: {
        delay: 5500,
        disableOnInteraction: false,
    },
    pagination: {
        el: ".swiper-pagination",
    },
});


AOS.init({
  duration:800,
  delay:400
});

//chatbot intregation started from here


// // Toggle chatbot visibility
// function toggleChat() {
//   const chatbotPopup = document.getElementById('chatbot-popup');
//   chatbotPopup.style.display = chatbotPopup.style.display === 'block' ? 'none' : 'block';
// }

// // Handle sending messages
// function sendMessage() {
//   const userInput = document.getElementById('user-input');
//   const userMessage = userInput.value.trim();
//   if (userMessage === '') return;

//   // Display user message
//   displayMessage(userMessage, 'user');

//   // Clear input field
//   userInput.value = '';

//   // Generate bot response
//   generateBotResponse(userMessage);
// }

// // Display a message in the chatbot body
// function displayMessage(message, sender) {
//   const chatbotBody = document.getElementById('chatbot-body');
//   const messageDiv = document.createElement('div');
//   messageDiv.classList.add('chatbot-message', sender);

//   // Profile icon for bot or user
//   const profileIcon = document.createElement('img');
//   profileIcon.classList.add('profile-icon');
//   profileIcon.src = sender === 'bot' ? './static/images/bot1.jpg' : './static/images/user.webp';
//   profileIcon.alt = sender === 'bot' ? './static/images/bot1.jpg' : './static/images/user.webp';

//   // Message text
//   const messageText = document.createElement('span');
//   messageText.textContent = message;

//   messageDiv.appendChild(profileIcon);
//   messageDiv.appendChild(messageText);
//   chatbotBody.appendChild(messageDiv);

//   // Scroll to the bottom
//   chatbotBody.scrollTop = chatbotBody.scrollHeight;
// }

// // Generate a simple bot response
// function generateBotResponse(userMessage) {
//   let botMessage = 'I am sorry, I didn\'t understand that.';

//   // Define basic responses
//   const responses = {
//       'hi': 'Hi there! How can I help you?',
//       'hello': 'Hi there! How can I help you?',
//       'how are you': 'I am a bot, but I\'m here to help you!',
//       'what is your name': 'I am your friendly chatbot.',
//       'bye': 'Goodbye! Have a nice day!',
//       'how can you help me': 'I can suggest and support you, in using the predictions.',
//   };

//   // Check if a response exists
//   const lowerCaseMessage = userMessage.toLowerCase();
//   if (responses[lowerCaseMessage]) {
//       botMessage = responses[lowerCaseMessage];
//   }

//   // Display bot message
//   setTimeout(() => displayMessage(botMessage, 'bot'), 500);
// }




document.addEventListener("DOMContentLoaded", () => {
  const modal = document.getElementById("modal");
  const readMoreBtn = document.getElementById("readMoreBtn");
  const closeModal = document.getElementById("closeModal");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn.addEventListener("click", () => {
      modal.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal.addEventListener("click", () => {
      modal.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal) {
          modal.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});

// JavaScript to Open and Close Message Box
function openMessageBox() {
    document.getElementById('message-box').style.display = 'block';
    document.body.classList.add('blur');
}

function closeMessageBox() {
    document.getElementById('message-box').style.display = 'none';
    document.body.classList.remove('blur');
}



//starts here//
document.addEventListener("DOMContentLoaded", () => {
  const modal1 = document.getElementById("modal1");
  const readMoreBtn1 = document.getElementById("readMoreBtn1");
  const closeModal1 = document.getElementById("closeModal1");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn1.addEventListener("click", () => {
      modal1.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal1.addEventListener("click", () => {
      modal1.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal1) {
          modal1.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});
//ends here//

//starts here//
document.addEventListener("DOMContentLoaded", () => {
  const modal2 = document.getElementById("modal2");
  const readMoreBtn2 = document.getElementById("readMoreBtn2");
  const closeModal2 = document.getElementById("closeModal2");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn2.addEventListener("click", () => {
      modal2.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal2.addEventListener("click", () => {
      modal2.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal2) {
          modal2.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});
//2 ends here//

//starts here//
document.addEventListener("DOMContentLoaded", () => {
  const modal3 = document.getElementById("modal3");
  const readMoreBtn3 = document.getElementById("readMoreBtn3");
  const closeModal3 = document.getElementById("closeModal3");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn3.addEventListener("click", () => {
      modal3.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal3.addEventListener("click", () => {
      modal3.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal3) {
          modal3.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});
//3 ends here//

//starts here//
document.addEventListener("DOMContentLoaded", () => {
  const modal4 = document.getElementById("modal4");
  const readMoreBtn4 = document.getElementById("readMoreBtn4");
  const closeModal4 = document.getElementById("closeModal4");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn4.addEventListener("click", () => {
      modal4.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal4.addEventListener("click", () => {
      modal4.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal4) {
          modal4.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});
//4 ends here//

//starts here//
document.addEventListener("DOMContentLoaded", () => {
  const modal5 = document.getElementById("modal5");
  const readMoreBtn5 = document.getElementById("readMoreBtn5");
  const closeModal5 = document.getElementById("closeModal5");

  // Show the modal when the "Read More" button is clicked
  readMoreBtn5.addEventListener("click", () => {
      modal5.style.display = "block";
      document.body.style.overflow = "hidden"; // Prevent background scrolling
  });

  // Close the modal when the close button is clicked
  closeModal5.addEventListener("click", () => {
      modal5.style.display = "none";
      document.body.style.overflow = "auto"; // Restore background scrolling
  });

  // Close the modal when clicking outside the modal content
  window.addEventListener("click", (event) => {
      if (event.target === modal5) {
          modal5.style.display = "none";
          document.body.style.overflow = "auto"; // Restore background scrolling
      }
  });
});
//5 ends here//