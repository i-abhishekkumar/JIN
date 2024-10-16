let i = 0;
let placeholder = "";
const texts = [
  "Case involving Land Disputes in Karnataka",
  "Case involving Theft of a Mobile Phone",
  "Case involving Defamation on Social Media",
  "Case Involving Trolling on Social Media",
];
let textIndex = 0;
const speed = 180;

function type() {
  if (i < texts[textIndex].length) {
    placeholder += texts[textIndex].charAt(i);
    document
      .getElementById("query")
      // .getElementsByClassName("email-id-placeholder")

      .setAttribute("placeholder", placeholder);
    i++;
    setTimeout(type, speed);
  } else {
    // Reset i and placeholder for the next text in the array
    i = 0;
    placeholder = "";
    textIndex = (textIndex + 1) % texts.length; // Loop back to the first text if reached the end
    setTimeout(type, speed); // Delay before typing the next text
  }
}

type();
// --------------file upload
// const form = document.querySelector("form"),
//   nextBtn = form.querySelector(".nextBtn"),
//   backBtn = form.querySelector(".backBtn"),
//   allInput = form.querySelectorAll(".first input");

// nextBtn.addEventListener("click", () => {
//   allInput.forEach((input) => {
//     if (input.value != "") {
//       form.classList.add("secActive");
//     } else {
//       form.classList.remove("secActive");
//     }
//   });
// });

// backBtn.addEventListener("click", () => form.classList.remove("secActive"));

var moreFields = document.getElementById("more-fields");
var showMoreBtn = document.getElementById("show-more-btn");
var isHidden = true;

showMoreBtn.addEventListener("click", function () {
  // var pdfForm = element.parentElement.querySelector(".pdf-form");
  // console.log("clicked");
  // if (pdfForm.style.maxHeight) {
  //   pdfForm.style.maxHeight = null;
  //   document.querySelector("#show-more-btn").innerText = "Show More";
  // } else {
  //   pdfForm.style.maxHeight = pdfForm.scrollHeight + "px";
  //   document.querySelector("#show-more-btn").innerText = "Show Less";
  // }
  if (isHidden) {
    moreFields.classList.remove("hidden");

    showMoreBtn.textContent = "Show Less";
  } else {
    moreFields.classList.add("hidden");
    showMoreBtn.textContent = "Show More";
  }
  isHidden = !isHidden;
});
